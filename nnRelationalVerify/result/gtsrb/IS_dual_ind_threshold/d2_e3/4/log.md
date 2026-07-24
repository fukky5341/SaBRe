## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 3600 seconds
Split limit: 100
Threshold: 29.323554093


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804)
1: (-27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839)
2: (-24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370)
3: (-30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026)
4: (-29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310)
5: (-30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021)
6: (-36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424)
7: (-35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427)
8: (-38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448)
9: (-31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176)
10: (-43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761)
11: (-42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975)
12: (-37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267)
13: (-40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344)
14: (-65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691)
15: (-36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355)
16: (-45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702)
17: (-65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044)
18: (-37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104)
19: (-31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634)
20: (-27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922)
21: (-38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606)
22: (-37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680)
23: (-31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860)
24: (-32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260)
25: (-29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138)
26: (-46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656)
27: (-38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689)
28: (-31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827)
29: (-38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564)
30: (-37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264)
31: (-36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420)
32: (-33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436)
33: (-54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724)
34: (-47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865)
35: (-43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463)
36: (-41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179)
37: (-61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194)
38: (-53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804)
39: (-62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511)
40: (-55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721)
41: (-35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751)
42: (-29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.81 + 73.56 = 76.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -29.3529070, upper bound: 29.3529070

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3136239, upper bound: 29.3498756
time: 67.37 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3136239, upper bound: 29.3503749
time: 82.46 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 149.95 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 149.95
Output dim: 2, lower bound: -29.3136239, upper bound: 29.3498756
IS_A2, status: Status.UNKNOWN, split count: 1, time: 149.95
Output dim: 2, lower bound: -29.3136239, upper bound: 29.3503749

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -46.2311020, 30.1737118, -46.3269386, 30.1964340, -76.4275360, 76.5006485
1: -27.6831894, 28.6672592, -27.7416058, 28.6856213, -56.3688049, 56.4088669
2: -24.5106277, 25.4348145, -24.5956917, 25.4527054, -49.9633255, 50.0305061
3: -30.6282845, 28.4859524, -30.7288380, 28.5096378, -59.1379242, 59.2147903
4: -29.0639324, 27.0908699, -29.1614685, 27.1125221, -56.1764450, 56.2523384
5: -30.1942825, 30.9242840, -30.2811756, 30.9463463, -61.1406097, 61.2054596
6: -36.3314590, 22.7706795, -36.3631363, 22.8314190, -59.1628799, 59.1338120
7: -35.5500145, 31.9322891, -35.6173248, 31.9550476, -67.5050659, 67.5496140
8: -38.3992462, 29.8532219, -38.4894562, 29.8766708, -68.2759171, 68.3426819
9: -31.7515831, 26.9218063, -31.8324184, 26.9588528, -58.7104340, 58.7542229
10: -43.7914429, 37.7190399, -43.8502197, 37.8116837, -81.6031265, 81.5692596
11: -42.3473701, 33.0952606, -42.3816490, 33.2695389, -75.6169128, 75.4769135
12: -37.6185265, 35.5738602, -37.6513672, 35.7337646, -73.3522797, 73.2252274
13: -40.0420151, 43.4310570, -40.1082344, 43.4712677, -83.5132828, 83.5392914
14: -65.8653870, 23.2943935, -65.9302597, 23.4248314, -89.2902069, 89.2246552
15: -36.6851578, 25.1171913, -36.7925835, 25.1426811, -61.8278389, 61.9097710
16: -45.7908897, 31.5736485, -45.8488922, 31.6687050, -77.4595947, 77.4225388
17: -65.6022949, 41.7706566, -65.6553116, 42.0185204, -107.6208191, 107.4259644
18: -37.8251152, 33.4482079, -37.8647575, 33.5592041, -71.3843231, 71.3129578
19: -31.7587700, 20.4123974, -31.7885876, 20.4983177, -52.2570877, 52.2009811
20: -27.2045765, 20.7657623, -27.2362595, 20.8364239, -48.0410004, 48.0020218
21: -38.1974220, 23.6177330, -38.2334976, 23.7386208, -61.9360428, 61.8512306
22: -37.3763275, 24.9505577, -37.4170609, 25.0542660, -62.4305954, 62.3676186
23: -31.4304905, 22.4067230, -31.4563198, 22.4771385, -53.9076309, 53.8630409
24: -32.4900284, 23.5849915, -32.5232010, 23.6289482, -56.1189728, 56.1081886
25: -29.8357105, 28.2048340, -29.8697186, 28.2603130, -58.0960236, 58.0745506
26: -46.5220337, 33.5067139, -46.5624237, 33.6600800, -80.1821136, 80.0691376
27: -38.4335136, 22.3045444, -38.4728622, 22.3765774, -60.8100891, 60.7774048
28: -31.6339169, 25.9012852, -31.6607361, 25.9735622, -57.6074677, 57.5620193
29: -38.8564262, 24.7493362, -38.8928642, 24.8916645, -63.7480927, 63.6421890
30: -37.5331688, 26.2718887, -37.5632858, 26.3812065, -63.9143677, 63.8351746
31: -36.0723648, 25.5193901, -36.1084061, 25.5925770, -61.6649323, 61.6277962
32: -33.1667557, 25.2297440, -33.2034760, 25.2864017, -58.4531555, 58.4332161
33: -54.7312202, 33.2450638, -54.8534355, 33.2777100, -88.0089264, 88.0984955
34: -47.3706665, 23.5776863, -47.4569855, 23.6038933, -70.9745636, 71.0346680
35: -43.2634239, 28.0517731, -43.3455429, 28.0708733, -71.3342896, 71.3973160
36: -41.4158592, 31.1933556, -41.4595337, 31.2385159, -72.6543655, 72.6528854
37: -60.9180145, 28.1857872, -60.9709816, 28.2357941, -89.1538086, 89.1567688
38: -53.0967026, 40.6801758, -53.1818657, 40.7120399, -93.8087463, 93.8620453
39: -62.1363487, 34.7681122, -62.2438049, 34.7926331, -96.9289856, 97.0119171
40: -54.8767090, 23.4062901, -54.9476395, 23.4282570, -78.3049622, 78.3539276
41: -35.4865417, 21.6732903, -35.5272102, 21.7252693, -57.2118073, 57.2005005
42: -29.7344437, 19.9933186, -29.7681217, 20.0689621, -49.8033981, 49.7614403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=258, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1753

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2807780, upper bound: 29.3478032
time: 66.93 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3126478, upper bound: 29.3488882
time: 66.40 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -46.4302940, 30.3163681, -46.3955994, 30.2111759, -76.6414719, 76.7119675
1: -27.8045044, 28.7355576, -27.7837753, 28.6968994, -56.5014038, 56.5193329
2: -24.6677361, 25.5680847, -24.6570854, 25.4636993, -50.1314240, 50.2251701
3: -30.8187580, 28.6495171, -30.8012848, 28.5238876, -59.3426437, 59.4508018
4: -29.2466602, 27.2379017, -29.2361279, 27.1265202, -56.3731766, 56.4740257
5: -30.3600311, 31.1030064, -30.3436375, 30.9598808, -61.3199005, 61.4466438
6: -36.4300728, 22.8665295, -36.3839378, 22.8584042, -59.2884674, 59.2504654
7: -35.6982651, 32.0031662, -35.6661758, 31.9672680, -67.6655197, 67.6693344
8: -38.5709686, 29.9573803, -38.5550842, 29.8903294, -68.4612961, 68.5124664
9: -31.9051304, 27.0509930, -31.8909721, 26.9820175, -58.8871384, 58.9419632
10: -43.9299355, 37.9399261, -43.8902664, 37.8749046, -81.8048401, 81.8301926
11: -42.6340981, 33.4040108, -42.4044456, 33.3943787, -76.0284729, 75.8084564
12: -37.8868828, 35.8626938, -37.6704178, 35.8490753, -73.7359619, 73.5331039
13: -40.1375580, 43.5608406, -40.1349449, 43.4974403, -83.6349945, 83.6957855
14: -66.1376877, 23.5262127, -65.9719009, 23.5190735, -89.6567535, 89.4981155
15: -36.8896637, 25.2123814, -36.8667107, 25.1597233, -62.0493851, 62.0790901
16: -45.9419441, 31.7512550, -45.8878250, 31.7169380, -77.6588821, 77.6390762
17: -66.0121231, 42.2142410, -65.6883087, 42.1998787, -108.2119904, 107.9025497
18: -38.0033607, 33.6522903, -37.8871765, 33.6388626, -71.6422272, 71.5394669
19: -31.9460506, 20.5704384, -31.8084641, 20.5607567, -52.5068054, 52.3788986
20: -27.3623199, 20.8951073, -27.2569122, 20.8868542, -48.2491760, 48.1520195
21: -38.4488335, 23.8324127, -38.2574615, 23.8260155, -62.2748413, 62.0898743
22: -37.5747833, 25.1489677, -37.4405098, 25.1304932, -62.7052765, 62.5894775
23: -31.5819378, 22.5312996, -31.4733372, 22.5264969, -54.1084290, 54.0046387
24: -32.6270447, 23.6712685, -32.5426903, 23.6598892, -56.2869339, 56.2139587
25: -29.9524403, 28.3156643, -29.8903904, 28.3010883, -58.2535248, 58.2060547
26: -46.8389015, 33.7768974, -46.5843163, 33.7705994, -80.6094971, 80.3612137
27: -38.6256256, 22.4340496, -38.4976196, 22.4292011, -61.0548248, 60.9316673
28: -31.8013630, 26.0314941, -31.6786785, 26.0253162, -57.8266754, 57.7101746
29: -39.1041641, 25.0006599, -38.9139557, 24.9965420, -64.1007004, 63.9146156
30: -37.7127304, 26.4684219, -37.5825691, 26.4581223, -64.1708527, 64.0509949
31: -36.2603683, 25.6570282, -36.1320190, 25.6459732, -61.9063339, 61.7890396
32: -33.2728424, 25.3321171, -33.2259064, 25.3189602, -58.5918045, 58.5580215
33: -54.9629555, 33.4308548, -54.9373436, 33.2986794, -88.2616272, 88.3681946
34: -47.5406189, 23.6959076, -47.5178719, 23.6212578, -71.1618729, 71.2137756
35: -43.4226265, 28.1297073, -43.4006424, 28.0826054, -71.5052185, 71.5303421
36: -41.5183716, 31.2932472, -41.4820061, 31.2715855, -72.7899551, 72.7752533
37: -61.0610771, 28.2854042, -61.0038147, 28.2647705, -89.3258514, 89.2892151
38: -53.2884483, 40.7716331, -53.2372055, 40.7313156, -94.0197601, 94.0088348
39: -62.3398361, 34.9335403, -62.3157425, 34.8071175, -97.1469574, 97.2492828
40: -55.0316200, 23.4878178, -54.9944420, 23.4412556, -78.4728775, 78.4822617
41: -35.5860100, 21.7613983, -35.5534401, 21.7470703, -57.3330803, 57.3148384
42: -29.8313961, 20.1396427, -29.7917328, 20.1241016, -49.9554977, 49.9313736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=258, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=479, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1753

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3176166, upper bound: 29.3483383
time: 83.49 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3493929, upper bound: 29.3493934
time: 63.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 149.29 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 149.29
Output dim: 2, lower bound: -29.2807780, upper bound: 29.3478032
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 149.29
Output dim: 2, lower bound: -29.3126478, upper bound: 29.3488882
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 149.29
Output dim: 2, lower bound: -29.3176166, upper bound: 29.3483383
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 149.29
Output dim: 2, lower bound: -29.3493929, upper bound: 29.3493934

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -46.1951714, 30.1313057, -46.2545776, 30.1112194, -76.3063889, 76.3858795
1: -27.6685543, 28.6173687, -27.7123013, 28.5867004, -56.2552567, 56.3296700
2: -24.4994717, 25.3736992, -24.5732880, 25.3320770, -49.8315506, 49.9469872
3: -30.6202831, 28.4213219, -30.7127838, 28.3805256, -59.0008087, 59.1341057
4: -29.0525990, 27.0225639, -29.1386757, 26.9746971, -56.0272903, 56.1612396
5: -30.1846638, 30.8677883, -30.2618961, 30.8346500, -61.0193100, 61.1296844
6: -36.2751083, 22.7560940, -36.2510567, 22.8019943, -59.0771027, 59.0071487
7: -35.5367699, 31.8772926, -35.5908051, 31.8454647, -67.3822327, 67.4680939
8: -38.3871422, 29.7445221, -38.4652481, 29.6572647, -68.0443954, 68.2097702
9: -31.7377090, 26.8976040, -31.8046074, 26.9108200, -58.6485291, 58.7022095
10: -43.7676697, 37.6994095, -43.8025169, 37.7724228, -81.5400848, 81.5019226
11: -42.2793083, 33.0774384, -42.2460785, 33.2335854, -75.5128937, 75.3235168
12: -37.5597992, 35.5576248, -37.5334091, 35.7010880, -73.2608871, 73.0910339
13: -40.0225067, 43.3747368, -40.0690765, 43.3586731, -83.3811798, 83.4438171
14: -65.8303070, 23.2525043, -65.8598099, 23.3413200, -89.1716156, 89.1123123
15: -36.6617584, 25.0821667, -36.7456818, 25.0726681, -61.7344208, 61.8278465
16: -45.7482300, 31.5573101, -45.7632675, 31.6359291, -77.3841476, 77.3205795
17: -65.5604324, 41.7501450, -65.5716553, 41.9773560, -107.5377808, 107.3218002
18: -37.7309570, 33.4370499, -37.6759529, 33.5368080, -71.2677612, 71.1129990
19: -31.7168713, 20.4056854, -31.7049236, 20.4848175, -52.2016830, 52.1106110
20: -27.1811142, 20.7460976, -27.1895447, 20.7968674, -47.9779816, 47.9356384
21: -38.1543159, 23.6064873, -38.1475601, 23.7160187, -61.8703346, 61.7540474
22: -37.3433914, 24.9385605, -37.3507233, 25.0302944, -62.3736877, 62.2892838
23: -31.3812447, 22.3921719, -31.3574123, 22.4477749, -53.8290176, 53.7495842
24: -32.4367561, 23.5763054, -32.4168930, 23.6115723, -56.0483284, 55.9931908
25: -29.8021259, 28.1914616, -29.8022690, 28.2336349, -58.0357590, 57.9937248
26: -46.4341354, 33.4898071, -46.3864517, 33.6261749, -80.0603027, 79.8762589
27: -38.3966179, 22.2928581, -38.3993225, 22.3529186, -60.7495346, 60.6921730
28: -31.5872345, 25.8857079, -31.5682487, 25.9422283, -57.5294647, 57.4539566
29: -38.8185806, 24.7397976, -38.8172455, 24.8725815, -63.6911621, 63.5570450
30: -37.4830093, 26.2515678, -37.4624634, 26.3405170, -63.8235245, 63.7140312
31: -36.0140610, 25.5098457, -35.9912910, 25.5735397, -61.5876007, 61.5011330
32: -33.1429558, 25.2124767, -33.1558533, 25.2514629, -58.3944168, 58.3683281
33: -54.6951714, 33.2283325, -54.7820396, 33.2442055, -87.9393768, 88.0103607
34: -47.3136597, 23.5643978, -47.3441887, 23.5773106, -70.8909683, 70.9085846
35: -43.2206497, 28.0405540, -43.2601395, 28.0484390, -71.2690887, 71.3006897
36: -41.3746376, 31.1856098, -41.3767776, 31.2229500, -72.5975876, 72.5623856
37: -60.8140755, 28.1761723, -60.7617416, 28.2164078, -89.0304871, 88.9379120
38: -53.0521584, 40.6665573, -53.0939789, 40.6846542, -93.7368164, 93.7605362
39: -62.1033859, 34.7538795, -62.1775513, 34.7641068, -96.8674927, 96.9314270
40: -54.8005333, 23.3912544, -54.7941742, 23.3981152, -78.1986465, 78.1854248
41: -35.4244843, 21.6597023, -35.4030075, 21.6978817, -57.1223602, 57.0627098
42: -29.7032604, 19.9757233, -29.7059364, 20.0335579, -49.7368164, 49.6816559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=476, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3150392
time: 85.28 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3462751
time: 64.97 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -46.2245598, 30.1675739, -46.4144211, 30.1995831, -76.4241409, 76.5819855
1: -27.6797581, 28.6628284, -27.8232098, 28.6907959, -56.3705521, 56.4860306
2: -24.5082932, 25.4293156, -24.6832561, 25.4560337, -49.9643250, 50.1125717
3: -30.6264687, 28.4784069, -30.8291874, 28.5114403, -59.1378937, 59.3075905
4: -29.0611000, 27.0833778, -29.2695122, 27.1172199, -56.1783218, 56.3528900
5: -30.1920509, 30.9186516, -30.3743858, 30.9508514, -61.1429024, 61.2930374
6: -36.3234711, 22.7670212, -36.3669472, 22.9190598, -59.2425079, 59.1339684
7: -35.5465240, 31.9228325, -35.6999130, 31.9549427, -67.5014648, 67.6227417
8: -38.3959045, 29.8411789, -38.6352081, 29.8753090, -68.2711945, 68.4763870
9: -31.7484436, 26.9174061, -31.8592453, 26.9676018, -58.7160454, 58.7766495
10: -43.7871895, 37.7157822, -43.8727684, 37.8620453, -81.6492310, 81.5885468
11: -42.3350830, 33.0909805, -42.3864403, 33.3353271, -75.6704102, 75.4774170
12: -37.6110001, 35.5708733, -37.6542091, 35.8350906, -73.4460907, 73.2250824
13: -40.0379562, 43.4237709, -40.1795502, 43.4895058, -83.5274582, 83.6033173
14: -65.8596649, 23.2878876, -66.0099640, 23.4353867, -89.2950516, 89.2978516
15: -36.6797714, 25.1047249, -36.8397903, 25.1417809, -61.8215523, 61.9445114
16: -45.7769547, 31.5708275, -45.8564415, 31.7380009, -77.5149384, 77.4272690
17: -65.5940247, 41.7667694, -65.6842499, 42.0484390, -107.6424637, 107.4510193
18: -37.8145180, 33.4459496, -37.8737259, 33.7122879, -71.5268097, 71.3196716
19: -31.7531376, 20.4108067, -31.7971478, 20.5441418, -52.2972794, 52.2079544
20: -27.1999073, 20.7617416, -27.2475204, 20.8775215, -48.0774231, 48.0092621
21: -38.1915321, 23.6150932, -38.2500496, 23.7725468, -61.9640732, 61.8651352
22: -37.3706093, 24.9466362, -37.4449844, 25.0707245, -62.4413300, 62.3916206
23: -31.4244308, 22.4039230, -31.4653149, 22.5331993, -53.9576263, 53.8692398
24: -32.4826355, 23.5825710, -32.5459595, 23.6862488, -56.1688843, 56.1285248
25: -29.8292370, 28.2006607, -29.8839245, 28.2982273, -58.1274567, 58.0845795
26: -46.5120392, 33.5026741, -46.5721436, 33.7723389, -80.2843781, 80.0748138
27: -38.4275360, 22.3024654, -38.4968948, 22.4052410, -60.8327789, 60.7993622
28: -31.6289825, 25.8985443, -31.6698704, 26.0344257, -57.6633911, 57.5684128
29: -38.8498726, 24.7466564, -38.9248314, 24.9222050, -63.7720795, 63.6714859
30: -37.5269012, 26.2671738, -37.5855446, 26.4687290, -63.9956284, 63.8527145
31: -36.0638962, 25.5168343, -36.1218300, 25.6422195, -61.7061157, 61.6386642
32: -33.1625328, 25.2267647, -33.2296448, 25.3432350, -58.5057678, 58.4564018
33: -54.7252617, 33.2407494, -54.8679886, 33.3321228, -88.0573883, 88.1087341
34: -47.3648300, 23.5743408, -47.4699020, 23.6822872, -71.0471191, 71.0442429
35: -43.2570496, 28.0489616, -43.3571739, 28.1190567, -71.3761063, 71.4061356
36: -41.4099426, 31.1914978, -41.4704590, 31.2860298, -72.6959686, 72.6619568
37: -60.9052429, 28.1837978, -60.9834824, 28.3788853, -89.2841187, 89.1672821
38: -53.0905952, 40.6775475, -53.2014732, 40.7666779, -93.8572693, 93.8790207
39: -62.1278305, 34.7646255, -62.2647362, 34.8052788, -96.9331055, 97.0293579
40: -54.8670120, 23.4023724, -54.9617004, 23.5533180, -78.4203339, 78.3640594
41: -35.4777679, 21.6706352, -35.5295258, 21.7997456, -57.2775116, 57.2001610
42: -29.7270298, 19.9892292, -29.7676487, 20.1381683, -49.8651962, 49.7568779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1609

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3163312
time: 82.72 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3112242, upper bound: 29.3474818
time: 72.45 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -46.3943672, 30.2739487, -46.3232841, 30.1259537, -76.5203171, 76.5972290
1: -27.7899055, 28.6856785, -27.7544746, 28.5979462, -56.3878479, 56.4401512
2: -24.6566029, 25.5069427, -24.6347084, 25.3430805, -49.9996834, 50.1416512
3: -30.8107700, 28.5848942, -30.7852402, 28.3947887, -59.2055588, 59.3701286
4: -29.2352982, 27.1696072, -29.2133083, 26.9887123, -56.2240105, 56.3829155
5: -30.3504143, 31.0464630, -30.3243790, 30.8482018, -61.1986084, 61.3708420
6: -36.3737106, 22.8519211, -36.2718506, 22.8289776, -59.2026901, 59.1237717
7: -35.6849899, 31.9482098, -35.6396141, 31.8577042, -67.5426865, 67.5878220
8: -38.5588760, 29.8486500, -38.5308914, 29.6709347, -68.2298126, 68.3795395
9: -31.8912506, 27.0267830, -31.8631458, 26.9340477, -58.8252869, 58.8899307
10: -43.9062157, 37.9203262, -43.8425980, 37.8356133, -81.7418213, 81.7629166
11: -42.5659790, 33.3861542, -42.2688866, 33.3584404, -75.9244232, 75.6550369
12: -37.8281593, 35.8464890, -37.5524483, 35.8164253, -73.6445847, 73.3989410
13: -40.1180687, 43.5045052, -40.0958214, 43.3848686, -83.5029373, 83.6003265
14: -66.1024475, 23.4843388, -65.9013977, 23.4356327, -89.5380783, 89.3857346
15: -36.8663063, 25.1773701, -36.8198891, 25.0897045, -61.9560089, 61.9972610
16: -45.8993034, 31.7349625, -45.8021774, 31.6841240, -77.5834122, 77.5371323
17: -65.9701920, 42.1937904, -65.6045685, 42.1586990, -108.1288910, 107.7983551
18: -37.9091759, 33.6411896, -37.6983910, 33.6164856, -71.5256653, 71.3395844
19: -31.9040966, 20.5636978, -31.7247963, 20.5472832, -52.4513779, 52.2884941
20: -27.3387833, 20.8754158, -27.2102413, 20.8472919, -48.1860733, 48.0856552
21: -38.4056816, 23.8211536, -38.1715393, 23.8033791, -62.2090607, 61.9926910
22: -37.5417671, 25.1369781, -37.3741608, 25.1065063, -62.6482735, 62.5111389
23: -31.5326443, 22.5167599, -31.3744354, 22.4971542, -54.0298004, 53.8911934
24: -32.5738144, 23.6625938, -32.4364090, 23.6425400, -56.2163544, 56.0990028
25: -29.9188538, 28.3022938, -29.8229561, 28.2744293, -58.1932755, 58.1252518
26: -46.7509460, 33.7599716, -46.4083900, 33.7366638, -80.4875946, 80.1683655
27: -38.5887070, 22.4223061, -38.4240417, 22.4055595, -60.9942627, 60.8463478
28: -31.7546349, 26.0159435, -31.5861797, 25.9939728, -57.7486076, 57.6021233
29: -39.0662994, 24.9911118, -38.8383751, 24.9774189, -64.0437164, 63.8294868
30: -37.6625519, 26.4480896, -37.4817581, 26.4174328, -64.0799866, 63.9298477
31: -36.2019730, 25.6474991, -36.0148849, 25.6269264, -61.8288956, 61.6623802
32: -33.2490273, 25.3147945, -33.1782684, 25.2840233, -58.5330505, 58.4930649
33: -54.9270287, 33.4140472, -54.8659668, 33.2651291, -88.1921539, 88.2800140
34: -47.4836884, 23.6825943, -47.4051361, 23.5946484, -71.0783310, 71.0877304
35: -43.3799400, 28.1184635, -43.3151970, 28.0601311, -71.4400635, 71.4336624
36: -41.4771805, 31.2854633, -41.3993301, 31.2559967, -72.7331772, 72.6847916
37: -60.9572220, 28.2757797, -60.7945328, 28.2453918, -89.2026062, 89.0703125
38: -53.2440147, 40.7579422, -53.1493149, 40.7038345, -93.9478455, 93.9072571
39: -62.3068428, 34.9192047, -62.2495346, 34.7785492, -97.0853729, 97.1687317
40: -54.9554520, 23.4728241, -54.8409576, 23.4111023, -78.3665543, 78.3137817
41: -35.5240402, 21.7477932, -35.4292374, 21.7196350, -57.2436676, 57.1770325
42: -29.8002396, 20.1220493, -29.7295322, 20.0887165, -49.8889542, 49.8515816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=479, inp2_unstable=476, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1609

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.3161032, upper bound: 29.3155712
time: 70.02 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3462751
time: 71.89 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -46.4237213, 30.3102360, -46.4830818, 30.2142715, -76.6379929, 76.7933121
1: -27.8010941, 28.7311020, -27.8653927, 28.7020607, -56.5031548, 56.5964966
2: -24.6654205, 25.5625896, -24.7446556, 25.4670143, -50.1324310, 50.3072433
3: -30.8169708, 28.6419964, -30.9016476, 28.5256710, -59.3426437, 59.5436440
4: -29.2438164, 27.2303829, -29.3441715, 27.1312256, -56.3750420, 56.5745506
5: -30.3578224, 31.0973587, -30.4368839, 30.9643478, -61.3221703, 61.5342369
6: -36.4221039, 22.8628941, -36.3877487, 22.9460564, -59.3681602, 59.2506409
7: -35.6947670, 31.9937668, -35.7487564, 31.9671574, -67.6619110, 67.7425232
8: -38.5676231, 29.9453449, -38.7008095, 29.8889122, -68.4565353, 68.6461563
9: -31.9020042, 27.0465965, -31.9177723, 26.9907837, -58.8927879, 58.9643631
10: -43.9257469, 37.9366455, -43.9128532, 37.9253731, -81.8511200, 81.8494949
11: -42.6217537, 33.3996887, -42.4092102, 33.4602013, -76.0819550, 75.8088989
12: -37.8793831, 35.8596992, -37.6732254, 35.9504471, -73.8298264, 73.5329208
13: -40.1334610, 43.5535355, -40.2062874, 43.5156937, -83.6491470, 83.7598190
14: -66.1319046, 23.5197487, -66.0515594, 23.5296917, -89.6615906, 89.5713043
15: -36.8842697, 25.1999264, -36.9139900, 25.1588364, -62.0431061, 62.1139145
16: -45.9280052, 31.7484360, -45.8953667, 31.7862740, -77.7142792, 77.6437988
17: -66.0038605, 42.2103424, -65.7171936, 42.2297554, -108.2336121, 107.9275360
18: -37.9927521, 33.6500740, -37.8961296, 33.7919388, -71.7846832, 71.5462036
19: -31.9404030, 20.5688629, -31.8170071, 20.6065903, -52.5469894, 52.3858719
20: -27.3576355, 20.8910542, -27.2682228, 20.9279461, -48.2855835, 48.1592789
21: -38.4429398, 23.8297424, -38.2740211, 23.8599815, -62.3029099, 62.1037636
22: -37.5690269, 25.1450462, -37.4684677, 25.1470032, -62.7160187, 62.6135063
23: -31.5758572, 22.5285034, -31.4823246, 22.5825882, -54.1584473, 54.0108261
24: -32.6196671, 23.6688652, -32.5654678, 23.7172089, -56.3368683, 56.2343292
25: -29.9459515, 28.3114700, -29.9045944, 28.3390541, -58.2850037, 58.2160645
26: -46.8288727, 33.7728844, -46.5940285, 33.8828888, -80.7117615, 80.3669128
27: -38.6196671, 22.4319420, -38.5216026, 22.4578552, -61.0775146, 60.9535408
28: -31.7964058, 26.0287857, -31.6877632, 26.0861835, -57.8825912, 57.7165489
29: -39.0975876, 24.9979973, -38.9459610, 25.0270481, -64.1246338, 63.9439583
30: -37.7064247, 26.4637012, -37.6048126, 26.5456600, -64.2520828, 64.0685120
31: -36.2518921, 25.6544743, -36.1454201, 25.6956406, -61.9475327, 61.7998810
32: -33.2686157, 25.3291245, -33.2520676, 25.3758316, -58.6444473, 58.5811920
33: -54.9569626, 33.4265556, -54.9519196, 33.3531303, -88.3100891, 88.3784714
34: -47.5347862, 23.6925583, -47.5308228, 23.6996307, -71.2344208, 71.2233810
35: -43.4163017, 28.1268711, -43.4122467, 28.1307793, -71.5470810, 71.5391083
36: -41.5124359, 31.2913570, -41.4929657, 31.3191071, -72.8315430, 72.7843170
37: -61.0483093, 28.2834167, -61.0162621, 28.4078560, -89.4561615, 89.2996826
38: -53.2823715, 40.7689514, -53.2567787, 40.7859497, -94.0683212, 94.0257263
39: -62.3313293, 34.9300232, -62.3367195, 34.8197937, -97.1511230, 97.2667389
40: -55.0219307, 23.4839172, -55.0084724, 23.5663300, -78.5882568, 78.4923859
41: -35.5772552, 21.7587318, -35.5557289, 21.8215408, -57.3987961, 57.3144569
42: -29.8239803, 20.1355782, -29.7912426, 20.1933651, -50.0173454, 49.9268188

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=479, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1609

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1751

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3480191, upper bound: 29.3168572
time: 225.36 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3480191, upper bound: 29.3480195
time: 65.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 293.49 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 293.49
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3150392
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 293.49
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3462751
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 293.49
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3163312
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 293.49
Output dim: 2, lower bound: -29.3112242, upper bound: 29.3474818
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 293.49
Output dim: 2, lower bound: -29.3161032, upper bound: 29.3155712
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 293.49
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3462751
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 293.49
Output dim: 2, lower bound: -29.3480191, upper bound: 29.3168572
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 293.49
Output dim: 2, lower bound: -29.3480191, upper bound: 29.3480195

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -46.1825905, 30.1214333, -46.2480316, 30.1059227, -76.2885132, 76.3694611
1: -27.6629066, 28.6077766, -27.7093639, 28.5812454, -56.2441483, 56.3171387
2: -24.4948425, 25.3635368, -24.5708942, 25.3265514, -49.8213959, 49.9344254
3: -30.6163387, 28.4064713, -30.7107563, 28.3729839, -58.9893188, 59.1172256
4: -29.0485611, 27.0105495, -29.1365623, 26.9685631, -56.0171242, 56.1471100
5: -30.1808701, 30.8583679, -30.2599030, 30.8294830, -61.0103531, 61.1182709
6: -36.2626495, 22.7495937, -36.2447510, 22.7986355, -59.0612869, 58.9943390
7: -35.5319710, 31.8658295, -35.5882797, 31.8395672, -67.3715363, 67.4541016
8: -38.3815308, 29.7234364, -38.4623032, 29.6464653, -68.0279846, 68.1857376
9: -31.7322311, 26.8908176, -31.8017216, 26.9073582, -58.6395874, 58.6925392
10: -43.7564621, 37.6944046, -43.7968521, 37.7697601, -81.5262146, 81.4912567
11: -42.2615852, 33.0712242, -42.2368889, 33.2303963, -75.4919815, 75.3081131
12: -37.5455475, 35.5523338, -37.5257797, 35.6983566, -73.2439041, 73.0781097
13: -40.0144234, 43.3631592, -40.0647964, 43.3527298, -83.3671417, 83.4279556
14: -65.8197937, 23.2449875, -65.8542328, 23.3375816, -89.1573639, 89.0992203
15: -36.6528740, 25.0662823, -36.7410126, 25.0639057, -61.7167740, 61.8072891
16: -45.7341309, 31.5522461, -45.7559128, 31.6332626, -77.3673935, 77.3081589
17: -65.5484772, 41.7258072, -65.5652695, 41.9639664, -107.5124435, 107.2910767
18: -37.7179375, 33.4316330, -37.6691360, 33.5340157, -71.2519531, 71.1007690
19: -31.7085972, 20.4031277, -31.7006474, 20.4834824, -52.1920776, 52.1037750
20: -27.1742897, 20.7387848, -27.1860085, 20.7930603, -47.9673500, 47.9247932
21: -38.1440163, 23.6020947, -38.1421089, 23.7137489, -61.8577576, 61.7442017
22: -37.3344231, 24.9206390, -37.3459549, 25.0211868, -62.3556061, 62.2665939
23: -31.3702755, 22.3882847, -31.3517628, 22.4457626, -53.8160324, 53.7400475
24: -32.4239426, 23.5722847, -32.4101868, 23.6094627, -56.0333939, 55.9824677
25: -29.7887974, 28.1860561, -29.7954254, 28.2307968, -58.0195923, 57.9814796
26: -46.4208488, 33.4813919, -46.3791466, 33.6218071, -80.0426559, 79.8605347
27: -38.3899918, 22.2823105, -38.3958054, 22.3475170, -60.7375031, 60.6781158
28: -31.5791912, 25.8808136, -31.5637951, 25.9396935, -57.5188751, 57.4446106
29: -38.8081017, 24.7261562, -38.8116760, 24.8656788, -63.6737823, 63.5378342
30: -37.4723053, 26.2440662, -37.4569550, 26.3366547, -63.8089447, 63.7010193
31: -35.9989853, 25.5052299, -35.9835815, 25.5710907, -61.5700760, 61.4888115
32: -33.1341705, 25.2070789, -33.1513290, 25.2486572, -58.3828125, 58.3584023
33: -54.6839676, 33.2228546, -54.7760544, 33.2413635, -87.9253311, 87.9989014
34: -47.3035126, 23.5585899, -47.3385353, 23.5742455, -70.8777618, 70.8971252
35: -43.2100410, 28.0364780, -43.2546501, 28.0462933, -71.2563248, 71.2911301
36: -41.3654938, 31.1814747, -41.3719254, 31.2207870, -72.5862808, 72.5533981
37: -60.7936974, 28.1724777, -60.7510757, 28.2145042, -89.0081940, 88.9235535
38: -53.0413475, 40.6608696, -53.0878258, 40.6815910, -93.7229309, 93.7486954
39: -62.0860825, 34.7492294, -62.1682053, 34.7615891, -96.8476562, 96.9174271
40: -54.7826271, 23.3852005, -54.7850609, 23.3949471, -78.1775742, 78.1702576
41: -35.4141808, 21.6555023, -35.3976746, 21.6956692, -57.1098480, 57.0531731
42: -29.6942425, 19.9696865, -29.7012920, 20.0303898, -49.7246323, 49.6709747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=476, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3058565
time: 79.24 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3058565
time: 67.89 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -46.2119751, 30.1577415, -46.4078979, 30.1942310, -76.4062042, 76.5656433
1: -27.6741142, 28.6532383, -27.8202763, 28.6853142, -56.3594284, 56.4735146
2: -24.5036564, 25.4191895, -24.6808681, 25.4504967, -49.9541550, 50.1000557
3: -30.6225147, 28.4635563, -30.8271790, 28.5038624, -59.1263771, 59.2907333
4: -29.0570831, 27.0713120, -29.2674160, 27.1111126, -56.1681976, 56.3387222
5: -30.1882706, 30.9092369, -30.3724194, 30.9456387, -61.1339111, 61.2816544
6: -36.3110123, 22.7605724, -36.3606186, 22.9157314, -59.2267303, 59.1211891
7: -35.5416985, 31.9113941, -35.6973648, 31.9490013, -67.4906998, 67.6087570
8: -38.3902664, 29.8200665, -38.6322861, 29.8644352, -68.2546997, 68.4523544
9: -31.7429428, 26.9105339, -31.8563538, 26.9640751, -58.7070084, 58.7668877
10: -43.7759628, 37.7107582, -43.8670883, 37.8594208, -81.6353760, 81.5778503
11: -42.3173294, 33.0848007, -42.3771820, 33.3321114, -75.6494293, 75.4619827
12: -37.5967751, 35.5656052, -37.6465759, 35.8323631, -73.4291382, 73.2121811
13: -40.0298538, 43.4121399, -40.1752625, 43.4835510, -83.5134048, 83.5874023
14: -65.8491592, 23.2802963, -66.0044022, 23.4315987, -89.2807617, 89.2846909
15: -36.6708832, 25.0888481, -36.8351593, 25.1330395, -61.8039093, 61.9240036
16: -45.7628250, 31.5657616, -45.8491020, 31.7353668, -77.4981842, 77.4148636
17: -65.5820618, 41.7425156, -65.6778641, 42.0349541, -107.6170044, 107.4203720
18: -37.8014679, 33.4405441, -37.8669090, 33.7094345, -71.5108948, 71.3074493
19: -31.7448635, 20.4082165, -31.7928505, 20.5427971, -52.2876587, 52.2010651
20: -27.1930847, 20.7544098, -27.2439804, 20.8737488, -48.0668335, 47.9983902
21: -38.1812439, 23.6106949, -38.2446060, 23.7703209, -61.9515533, 61.8553009
22: -37.3616333, 24.9287300, -37.4402237, 25.0616474, -62.4232788, 62.3689537
23: -31.4134178, 22.4000435, -31.4596481, 22.5311584, -53.9445648, 53.8596916
24: -32.4698257, 23.5785408, -32.5392532, 23.6841431, -56.1539612, 56.1177940
25: -29.8159046, 28.1952686, -29.8770390, 28.2953739, -58.1112747, 58.0723038
26: -46.4988136, 33.4942894, -46.5648270, 33.7680740, -80.2668915, 80.0591125
27: -38.4208908, 22.2919197, -38.4933472, 22.3998642, -60.8207550, 60.7852631
28: -31.6209583, 25.8937073, -31.6654148, 26.0319023, -57.6528625, 57.5591164
29: -38.8393135, 24.7330627, -38.9192162, 24.9153214, -63.7546349, 63.6522789
30: -37.5161362, 26.2597008, -37.5800018, 26.4648952, -63.9810333, 63.8397026
31: -36.0488243, 25.5122108, -36.1141052, 25.6397934, -61.6886177, 61.6263161
32: -33.1537247, 25.2213573, -33.2251282, 25.3404903, -58.4942093, 58.4464836
33: -54.7140007, 33.2352676, -54.8620796, 33.3292542, -88.0432587, 88.0973511
34: -47.3547211, 23.5685463, -47.4642906, 23.6792507, -71.0339737, 71.0328369
35: -43.2464752, 28.0449142, -43.3517075, 28.1169434, -71.3634186, 71.3966217
36: -41.4007683, 31.1873703, -41.4656487, 31.2838783, -72.6846466, 72.6530151
37: -60.8848343, 28.1801720, -60.9727783, 28.3769836, -89.2618103, 89.1529541
38: -53.0797691, 40.6718445, -53.1953468, 40.7636604, -93.8434296, 93.8671875
39: -62.1107140, 34.7599716, -62.2553940, 34.8027954, -96.9135132, 97.0153656
40: -54.8490601, 23.3962879, -54.9526100, 23.5502281, -78.3992844, 78.3488922
41: -35.4674492, 21.6664619, -35.5241394, 21.7975502, -57.2649994, 57.1906013
42: -29.7180042, 19.9832001, -29.7630348, 20.1350403, -49.8530426, 49.7462349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.3069861, upper bound: 29.3069979
time: 80.20 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3074997, upper bound: 29.3446372
time: 76.15 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -46.3817902, 30.2640839, -46.3167229, 30.1206322, -76.5024109, 76.5808105
1: -27.7842464, 28.6760979, -27.7515240, 28.5925255, -56.3767700, 56.4276161
2: -24.6519527, 25.4967785, -24.6323223, 25.3375740, -49.9895248, 50.1290970
3: -30.8068104, 28.5700283, -30.7832241, 28.3872433, -59.1940460, 59.3532524
4: -29.2312775, 27.1575661, -29.2111988, 26.9825840, -56.2138596, 56.3687668
5: -30.3466396, 31.0370655, -30.3223839, 30.8429966, -61.1896362, 61.3594475
6: -36.3612900, 22.8454571, -36.2655563, 22.8256283, -59.1869202, 59.1110153
7: -35.6801834, 31.9367046, -35.6370773, 31.8518028, -67.5319824, 67.5737839
8: -38.5532303, 29.8275909, -38.5279198, 29.6600971, -68.2133255, 68.3555145
9: -31.8857479, 27.0200081, -31.8602638, 26.9305801, -58.8163300, 58.8802719
10: -43.8950806, 37.9153404, -43.8368835, 37.8329659, -81.7280426, 81.7522278
11: -42.5482750, 33.3799629, -42.2596855, 33.3552170, -75.9034729, 75.6396484
12: -37.8138809, 35.8412247, -37.5448303, 35.8136787, -73.6275635, 73.3860550
13: -40.1099625, 43.4928932, -40.0915451, 43.3790092, -83.4889679, 83.5844269
14: -66.0919952, 23.4768925, -65.8958511, 23.4318962, -89.5238953, 89.3727417
15: -36.8574028, 25.1614895, -36.8152466, 25.0809250, -61.9383240, 61.9767342
16: -45.8851967, 31.7298660, -45.7948151, 31.6814461, -77.5666351, 77.5246811
17: -65.9582520, 42.1693954, -65.5982361, 42.1452751, -108.1035156, 107.7676315
18: -37.8961525, 33.6357498, -37.6915970, 33.6136475, -71.5097961, 71.3273468
19: -31.8958359, 20.5611572, -31.7205257, 20.5459404, -52.4417763, 52.2816849
20: -27.3319664, 20.8681126, -27.2066879, 20.8434906, -48.1754570, 48.0747986
21: -38.3953552, 23.8167534, -38.1661148, 23.8011169, -62.1964722, 61.9828682
22: -37.5328217, 25.1190586, -37.3693695, 25.0974121, -62.6302338, 62.4884262
23: -31.5216827, 22.5128651, -31.3687592, 22.4951267, -54.0168076, 53.8816223
24: -32.5610046, 23.6585579, -32.4296951, 23.6404305, -56.2014313, 56.0882530
25: -29.9055138, 28.2969246, -29.8161221, 28.2715874, -58.1771011, 58.1130447
26: -46.7376938, 33.7516022, -46.4010544, 33.7323151, -80.4700089, 80.1526566
27: -38.5820694, 22.4118214, -38.4205093, 22.4001827, -60.9822540, 60.8323288
28: -31.7466087, 26.0110741, -31.5817089, 25.9914589, -57.7380676, 57.5927811
29: -39.0557976, 24.9774895, -38.8327942, 24.9705296, -64.0263290, 63.8102837
30: -37.6518707, 26.4406166, -37.4762344, 26.4135838, -64.0654526, 63.9168510
31: -36.1869278, 25.6428833, -36.0072021, 25.6244698, -61.8113976, 61.6500854
32: -33.2402534, 25.3093967, -33.1737061, 25.2811966, -58.5214500, 58.4831009
33: -54.9157791, 33.4085388, -54.8600388, 33.2622528, -88.1780319, 88.2685776
34: -47.4735794, 23.6767693, -47.3995018, 23.5915852, -71.0651627, 71.0762634
35: -43.3693466, 28.1144180, -43.3097687, 28.0579987, -71.4273453, 71.4241791
36: -41.4680405, 31.2813683, -41.3944817, 31.2538681, -72.7219086, 72.6758499
37: -60.9368973, 28.2721252, -60.7839088, 28.2434883, -89.1803894, 89.0560303
38: -53.2331696, 40.7522507, -53.1432114, 40.7008438, -93.9340057, 93.8954620
39: -62.2895355, 34.9145851, -62.2401695, 34.7760468, -97.0655670, 97.1547546
40: -54.9375534, 23.4667797, -54.8318672, 23.4079208, -78.3454742, 78.2986450
41: -35.5137405, 21.7435780, -35.4239235, 21.7174454, -57.2311859, 57.1675034
42: -29.7912216, 20.1160316, -29.7249031, 20.0855618, -49.8767853, 49.8409309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=479, inp2_unstable=476, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1771

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3067859
time: 75.27 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3136875, upper bound: 29.3444516
time: 78.71 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -46.2472992, 30.2012100, -46.4347878, 30.1640530, -76.4113312, 76.6359940
1: -27.6893024, 28.5916004, -27.8473949, 28.6357651, -56.3250656, 56.4389954
2: -24.5557137, 25.4150867, -24.7302189, 25.3976288, -49.9533386, 50.1453056
3: -30.6984730, 28.4489613, -30.8914719, 28.4349823, -59.1334534, 59.3404312
4: -29.1282158, 27.0732174, -29.3310852, 27.0593185, -56.1875343, 56.4043007
5: -30.2493286, 30.9545898, -30.4247837, 30.8987045, -61.1480255, 61.3793716
6: -36.2670174, 22.7522678, -36.3176003, 22.9287567, -59.1957741, 59.0698700
7: -35.5820274, 31.8351517, -35.7347336, 31.8925285, -67.4745560, 67.5698853
8: -38.4322891, 29.6788197, -38.6873398, 29.7639465, -68.1962280, 68.3661499
9: -31.8448620, 26.9774132, -31.9009953, 26.9638176, -58.8086777, 58.8784103
10: -43.8156433, 37.8442421, -43.8739281, 37.9059296, -81.7215576, 81.7181702
11: -42.4846306, 33.3429871, -42.3550224, 33.4424362, -75.9270630, 75.6980133
12: -37.7123413, 35.7397690, -37.5963402, 35.9314690, -73.6438141, 73.3361053
13: -40.0284386, 43.4064636, -40.1850586, 43.4513397, -83.4797821, 83.5915222
14: -66.0064316, 23.4320393, -66.0112152, 23.4969025, -89.5033340, 89.4432526
15: -36.7585297, 25.0780087, -36.8896294, 25.1058578, -61.8643799, 61.9676361
16: -45.7770233, 31.6608467, -45.8347473, 31.7712612, -77.5482788, 77.4955902
17: -65.8675842, 42.1030197, -65.6733170, 42.1898880, -108.0574722, 107.7763367
18: -37.7663651, 33.4870377, -37.7940979, 33.7784424, -71.5448074, 71.2811356
19: -31.8296165, 20.5209694, -31.7703381, 20.5996552, -52.4292717, 52.2912979
20: -27.2867851, 20.8148193, -27.2390518, 20.9046307, -48.1914139, 48.0538712
21: -38.3341522, 23.7865524, -38.2313080, 23.8470078, -62.1811523, 62.0178528
22: -37.4470406, 25.0901718, -37.4330444, 25.1252480, -62.5722885, 62.5232162
23: -31.4775734, 22.4774933, -31.4407425, 22.5681114, -54.0456848, 53.9182358
24: -32.4683075, 23.6199493, -32.5042648, 23.7091465, -56.1774445, 56.1242142
25: -29.8413315, 28.2424622, -29.8632088, 28.3252182, -58.1665344, 58.1056671
26: -46.6942444, 33.6785355, -46.5345879, 33.8603172, -80.5545654, 80.2131195
27: -38.5291557, 22.3743019, -38.4907608, 22.4373589, -60.9665146, 60.8650551
28: -31.7103214, 25.9650517, -31.6501160, 26.0682030, -57.7785187, 57.6151657
29: -38.9561234, 24.9268913, -38.9059334, 24.9962616, -63.9523849, 63.8328247
30: -37.6036072, 26.3738384, -37.5660667, 26.5227108, -64.1263199, 63.9399033
31: -36.0491714, 25.5682640, -36.0568390, 25.6856499, -61.7348213, 61.6250954
32: -33.1800385, 25.2476921, -33.2211914, 25.3578892, -58.5379181, 58.4688835
33: -54.8244591, 33.3386993, -54.8938866, 33.3386917, -88.1631470, 88.2325821
34: -47.4001999, 23.6024647, -47.4696655, 23.6843014, -71.0845032, 71.0721283
35: -43.2957993, 28.0588493, -43.3583946, 28.1174679, -71.4132690, 71.4172363
36: -41.4331551, 31.2485466, -41.4598160, 31.3087654, -72.7419205, 72.7083511
37: -60.7876854, 28.1455917, -60.9012871, 28.3968296, -89.1845169, 89.0468750
38: -53.1401825, 40.6717224, -53.1968231, 40.7676392, -93.9078217, 93.8685455
39: -62.1755943, 34.8719025, -62.2738647, 34.8079224, -96.9835205, 97.1457672
40: -54.7861900, 23.3222351, -54.9061127, 23.5494232, -78.3356094, 78.2283401
41: -35.4486923, 21.7007694, -35.4979095, 21.8077583, -57.2564392, 57.1986771
42: -29.7464161, 20.0611362, -29.7574024, 20.1746502, -49.9210663, 49.8185387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=479, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3449520, upper bound: 29.2767014
time: 75.22 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3456274, upper bound: 29.3144508
time: 75.53 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -46.4111633, 30.3003750, -46.4765854, 30.2089672, -76.6201324, 76.7769623
1: -27.7954407, 28.7215652, -27.8624439, 28.6965828, -56.4920235, 56.5840073
2: -24.6607742, 25.5524521, -24.7422791, 25.4615021, -50.1222687, 50.2947311
3: -30.8130264, 28.6271038, -30.8996429, 28.5181026, -59.3311310, 59.5267487
4: -29.2397976, 27.2183666, -29.3420773, 27.1250706, -56.3648682, 56.5604401
5: -30.3540058, 31.0879364, -30.4349194, 30.9591751, -61.3131714, 61.5228577
6: -36.4096298, 22.8564243, -36.3814240, 22.9427109, -59.3523407, 59.2378464
7: -35.6899567, 31.9822655, -35.7462082, 31.9612255, -67.6511841, 67.7284622
8: -38.5619965, 29.9242134, -38.6979218, 29.8780632, -68.4400635, 68.6221313
9: -31.8965149, 27.0397491, -31.9148865, 26.9872818, -58.8837967, 58.9546356
10: -43.9145203, 37.9316254, -43.9071655, 37.9227257, -81.8372498, 81.8387833
11: -42.6040497, 33.3935356, -42.3999634, 33.4569397, -76.0609894, 75.7935028
12: -37.8651276, 35.8544540, -37.6656075, 35.9476776, -73.8128052, 73.5200577
13: -40.1253548, 43.5419159, -40.2020493, 43.5097580, -83.6351166, 83.7439651
14: -66.1214142, 23.5121346, -66.0460358, 23.5258389, -89.6472473, 89.5581665
15: -36.8753510, 25.1840305, -36.9093971, 25.1500645, -62.0254059, 62.0934258
16: -45.9138603, 31.7433586, -45.8879547, 31.7836018, -77.6974487, 77.6313095
17: -65.9919281, 42.1861267, -65.7108459, 42.2163010, -108.2082291, 107.8969650
18: -37.9797249, 33.6446304, -37.8893280, 33.7891159, -71.7688293, 71.5339584
19: -31.9321442, 20.5662575, -31.8127098, 20.6052513, -52.5373955, 52.3789673
20: -27.3508186, 20.8837395, -27.2646790, 20.9241924, -48.2750092, 48.1484184
21: -38.4326172, 23.8253479, -38.2685699, 23.8576832, -62.2902985, 62.0939178
22: -37.5600510, 25.1271362, -37.4637032, 25.1379089, -62.6979599, 62.5908356
23: -31.5648575, 22.5246162, -31.4766426, 22.5805492, -54.1454010, 54.0012589
24: -32.6068497, 23.6648197, -32.5587463, 23.7151146, -56.3219528, 56.2235641
25: -29.9326382, 28.3060856, -29.8977127, 28.3361950, -58.2688332, 58.2037964
26: -46.8156471, 33.7645264, -46.5866661, 33.8785858, -80.6942291, 80.3511963
27: -38.6129951, 22.4213886, -38.5180664, 22.4524879, -61.0654831, 60.9394531
28: -31.7883778, 26.0239105, -31.6833038, 26.0836601, -57.8720398, 57.7072067
29: -39.0870323, 24.9843712, -38.9403343, 25.0202103, -64.1072388, 63.9247055
30: -37.6956940, 26.4562130, -37.5992775, 26.5418034, -64.2374878, 64.0554886
31: -36.2368240, 25.6498222, -36.1376915, 25.6932068, -61.9300308, 61.7875099
32: -33.2598114, 25.3237381, -33.2475281, 25.3730659, -58.6328773, 58.5712624
33: -54.9457741, 33.4210815, -54.9459953, 33.3502808, -88.2960510, 88.3670731
34: -47.5246582, 23.6867466, -47.5251770, 23.6965771, -71.2212372, 71.2119217
35: -43.4056854, 28.1228371, -43.4067955, 28.1286621, -71.5343399, 71.5296326
36: -41.5032997, 31.2872372, -41.4881210, 31.3169937, -72.8202896, 72.7753601
37: -61.0279541, 28.2797527, -61.0055771, 28.4059677, -89.4339218, 89.2853241
38: -53.2714882, 40.7632446, -53.2506256, 40.7829285, -94.0544128, 94.0138702
39: -62.3141136, 34.9253349, -62.3273773, 34.8173141, -97.1314240, 97.2527161
40: -55.0039635, 23.4778690, -54.9993706, 23.5631943, -78.5671539, 78.4772339
41: -35.5669365, 21.7545395, -35.5503616, 21.8193512, -57.3862839, 57.3049011
42: -29.8149757, 20.1295261, -29.7866325, 20.1902313, -50.0052032, 49.9161568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=479, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3449520, upper bound: 29.3079366
time: 80.52 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3456274, upper bound: 29.3456277
time: 70.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 153.66 seconds
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 153.66
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3058565
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 153.66
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3058565
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 153.66
Output dim: 2, lower bound: -29.3069861, upper bound: 29.3069979
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 153.66
Output dim: 2, lower bound: -29.3074997, upper bound: 29.3446372
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 153.66
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3067859
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 153.66
Output dim: 2, lower bound: -29.3136875, upper bound: 29.3444516
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 153.66
Output dim: 2, lower bound: -29.3449520, upper bound: 29.2767014
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 153.66
Output dim: 2, lower bound: -29.3456274, upper bound: 29.3144508
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 153.66
Output dim: 2, lower bound: -29.3449520, upper bound: 29.3079366
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 153.66
Output dim: 2, lower bound: -29.3456274, upper bound: 29.3456277

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -46.2021866, 30.1553078, -46.4134483, 30.2703705, -76.4725494, 76.5687561
1: -27.6682129, 28.6510677, -27.8272324, 28.7113266, -56.3795395, 56.4782982
2: -24.4954891, 25.4170780, -24.6790848, 25.5167542, -50.0122452, 50.0961609
3: -30.6125488, 28.4603252, -30.8240814, 28.5957279, -59.2082748, 59.2844086
4: -29.0472641, 27.0684776, -29.2622013, 27.1924553, -56.2397156, 56.3306770
5: -30.1793709, 30.9061127, -30.3714237, 31.0475769, -61.2269440, 61.2775307
6: -36.3066101, 22.7468281, -36.3848343, 22.9096279, -59.2162323, 59.1316605
7: -35.5321121, 31.9084969, -35.7066269, 31.9763832, -67.5084915, 67.6151199
8: -38.3806305, 29.8164463, -38.6304092, 29.9346771, -68.3153076, 68.4468536
9: -31.7370377, 26.9032001, -31.8602409, 26.9893379, -58.7263718, 58.7634354
10: -43.7686043, 37.7003174, -43.8883057, 37.8806038, -81.6492081, 81.5886230
11: -42.3130531, 33.0710182, -42.5033531, 33.3233719, -75.6364288, 75.5743713
12: -37.5922089, 35.5489426, -37.8299065, 35.8223419, -73.4145508, 73.3788452
13: -40.0223007, 43.3999481, -40.1835251, 43.4876137, -83.5099106, 83.5834732
14: -65.8421478, 23.2710304, -66.0779877, 23.4294052, -89.2715530, 89.3490143
15: -36.6560249, 25.0858479, -36.8328094, 25.1673908, -61.8234100, 61.9186554
16: -45.7485123, 31.5454445, -45.8644028, 31.7324142, -77.4809265, 77.4098511
17: -65.5755157, 41.7243843, -65.8550110, 42.0231476, -107.5986481, 107.5793839
18: -37.7952271, 33.4325485, -37.8952255, 33.7119217, -71.5071487, 71.3277740
19: -31.7414570, 20.4018383, -31.8706150, 20.5413666, -52.2828217, 52.2724533
20: -27.1892548, 20.7485504, -27.2993870, 20.8731880, -48.0624428, 48.0479317
21: -38.1772804, 23.6019211, -38.3473358, 23.7687149, -61.9459915, 61.9492569
22: -37.3555374, 24.9218063, -37.4822235, 25.0694427, -62.4249802, 62.4040260
23: -31.4104843, 22.3948364, -31.5127411, 22.5311089, -53.9415894, 53.9075775
24: -32.4635849, 23.5755081, -32.5539551, 23.6893005, -56.1528854, 56.1294556
25: -29.8114510, 28.1885681, -29.9016819, 28.3009224, -58.1123734, 58.0902481
26: -46.4930038, 33.4816666, -46.7001343, 33.7606125, -80.2536163, 80.1818008
27: -38.4161072, 22.2852650, -38.5144653, 22.3983574, -60.8144646, 60.7997284
28: -31.6180096, 25.8888893, -31.7205162, 26.0323162, -57.6503258, 57.6094055
29: -38.8331795, 24.7224045, -39.0009804, 24.9082565, -63.7414360, 63.7233849
30: -37.5121841, 26.2519608, -37.6184616, 26.4664936, -63.9786758, 63.8704224
31: -36.0448036, 25.5068073, -36.1653976, 25.6424809, -61.6872864, 61.6722031
32: -33.1494064, 25.2133656, -33.2694016, 25.3394451, -58.4888535, 58.4827614
33: -54.7030563, 33.2305107, -54.8711662, 33.3804321, -88.0834808, 88.1016769
34: -47.3463669, 23.5648499, -47.4684067, 23.7438660, -71.0902328, 71.0332489
35: -43.2381058, 28.0416451, -43.3569145, 28.1403408, -71.3784485, 71.3985596
36: -41.3967781, 31.1812286, -41.4899597, 31.2898636, -72.6866379, 72.6711884
37: -60.8774796, 28.1736984, -61.0016174, 28.3869610, -89.2644348, 89.1753082
38: -53.0688438, 40.6656265, -53.2169724, 40.7794571, -93.8482971, 93.8825989
39: -62.1017990, 34.7510872, -62.2691498, 34.8218498, -96.9236450, 97.0202332
40: -54.8390732, 23.3922119, -54.9625320, 23.5723133, -78.4113846, 78.3547440
41: -35.4633980, 21.6546021, -35.5426407, 21.7927513, -57.2561493, 57.1972427
42: -29.7145958, 19.9741592, -29.7901669, 20.1332684, -49.8478622, 49.7643242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=479, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1609

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.3042600, upper bound: 29.3095815
time: 73.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3060802, upper bound: 29.3432088
time: 75.81 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -46.3726425, 30.2620239, -46.3229332, 30.1969013, -76.5695419, 76.5849609
1: -27.7786503, 28.6742706, -27.7589378, 28.6189194, -56.3975677, 56.4332085
2: -24.6443253, 25.4948788, -24.6312065, 25.4039841, -50.0483017, 50.1260834
3: -30.7973709, 28.5671558, -30.7808189, 28.4794426, -59.2768097, 59.3479691
4: -29.2221375, 27.1550045, -29.2069473, 27.0641937, -56.2863312, 56.3619537
5: -30.3381500, 31.0341969, -30.3218498, 30.9452343, -61.2833862, 61.3560486
6: -36.3571548, 22.8343582, -36.2899742, 22.8223724, -59.1795273, 59.1243248
7: -35.6713142, 31.9342937, -35.6469269, 31.8795624, -67.5508728, 67.5812149
8: -38.5441399, 29.8242683, -38.5267982, 29.7306423, -68.2747803, 68.3510590
9: -31.8803501, 27.0130444, -31.8647156, 26.9560738, -58.8364258, 58.8777580
10: -43.8886185, 37.9054604, -43.8591995, 37.8548279, -81.7434464, 81.7646561
11: -42.5443344, 33.3672638, -42.3861084, 33.3479424, -75.8922729, 75.7533646
12: -37.8097649, 35.8255081, -37.7285042, 35.8049278, -73.6146774, 73.5540085
13: -40.1030273, 43.4810257, -40.1004829, 43.3835220, -83.4865494, 83.5815048
14: -66.0857086, 23.4682903, -65.9703903, 23.4305916, -89.5162964, 89.4386826
15: -36.8439713, 25.1586361, -36.8144836, 25.1155109, -61.9594727, 61.9731140
16: -45.8716888, 31.7104111, -45.8106194, 31.6799202, -77.5515900, 77.5210266
17: -65.9521866, 42.1525192, -65.7758331, 42.1350479, -108.0872192, 107.9283524
18: -37.8901978, 33.6283569, -37.7202415, 33.6168022, -71.5070038, 71.3485947
19: -31.8927803, 20.5551910, -31.7985611, 20.5450459, -52.4378281, 52.3537445
20: -27.3286057, 20.8627110, -27.2623997, 20.8434772, -48.1720810, 48.1251106
21: -38.3917274, 23.8086433, -38.2690659, 23.8003807, -62.1921082, 62.0777092
22: -37.5274620, 25.1125526, -37.4119644, 25.1057968, -62.6332550, 62.5245094
23: -31.5191021, 22.5080795, -31.4221535, 22.4956512, -54.0147552, 53.9302330
24: -32.5552368, 23.6558590, -32.4451065, 23.6459579, -56.2011948, 56.1009598
25: -29.9014606, 28.2915936, -29.8413582, 28.2777252, -58.1791840, 58.1329460
26: -46.7327042, 33.7395935, -46.5371399, 33.7257462, -80.4584503, 80.2767334
27: -38.5776978, 22.4062767, -38.4421387, 22.3997536, -60.9774475, 60.8484154
28: -31.7439060, 26.0066357, -31.6370373, 25.9923000, -57.7362061, 57.6436729
29: -39.0503654, 24.9674721, -38.9152794, 24.9643211, -64.0146790, 63.8827515
30: -37.6481819, 26.4335690, -37.5149460, 26.4160957, -64.0642776, 63.9485054
31: -36.1832504, 25.6380501, -36.0588150, 25.6279125, -61.8111649, 61.6968651
32: -33.2362366, 25.3019505, -33.2182732, 25.2809258, -58.5171509, 58.5202255
33: -54.9056969, 33.4042931, -54.8701706, 33.3138847, -88.2195816, 88.2744598
34: -47.4657860, 23.6735516, -47.4042664, 23.6566448, -71.1224213, 71.0778198
35: -43.3616600, 28.1112823, -43.3157730, 28.0816269, -71.4432831, 71.4270477
36: -41.4645157, 31.2754002, -41.4193649, 31.2602634, -72.7247772, 72.6947632
37: -60.9302635, 28.2659073, -60.8134117, 28.2539883, -89.1842499, 89.0793152
38: -53.2233047, 40.7463226, -53.1665115, 40.7169952, -93.9403000, 93.9128342
39: -62.2814827, 34.9058304, -62.2549324, 34.7955627, -97.0770416, 97.1607590
40: -54.9284134, 23.4630642, -54.8423386, 23.4305840, -78.3589935, 78.3054047
41: -35.5100098, 21.7336369, -35.4427414, 21.7152367, -57.2252464, 57.1763725
42: -29.7881031, 20.1077747, -29.7522907, 20.0846157, -49.8727188, 49.8600655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=479, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1609

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.3102398, upper bound: 29.3091590
time: 64.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2717088, upper bound: 29.3420240
time: 92.20 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -46.1678581, 30.1858711, -46.2732086, 30.1328964, -76.3007507, 76.4590759
1: -27.6394749, 28.5792274, -27.7478733, 28.6108837, -56.2503586, 56.3270950
2: -24.4863205, 25.4033566, -24.5894318, 25.3737736, -49.8600922, 49.9927902
3: -30.6123581, 28.4317207, -30.7165012, 28.4000530, -59.0124130, 59.1482201
4: -29.0444088, 27.0586033, -29.1611767, 27.0295582, -56.0739670, 56.2197800
5: -30.1742649, 30.9382019, -30.2719212, 30.8653965, -61.0396500, 61.2101173
6: -36.2441177, 22.6928444, -36.2712288, 22.8102112, -59.0543289, 58.9640732
7: -35.5236664, 31.8196640, -35.6158905, 31.8615437, -67.3852081, 67.4355469
8: -38.3451729, 29.6619720, -38.5103836, 29.7298412, -68.0750122, 68.1723480
9: -31.7962074, 26.9459209, -31.8020420, 26.8990803, -58.6952896, 58.7479630
10: -43.7802925, 37.7646408, -43.8028259, 37.7446556, -81.5249481, 81.5674591
11: -42.4595604, 33.2189636, -42.3038025, 33.1905479, -75.6501007, 75.5227661
12: -37.6903687, 35.5888557, -37.5516129, 35.6243896, -73.3147583, 73.1404648
13: -40.0044289, 43.3658905, -40.1369057, 43.3690033, -83.3734283, 83.5027924
14: -65.9622345, 23.3483639, -65.9211884, 23.3280125, -89.2902451, 89.2695541
15: -36.6666336, 25.0597458, -36.7043610, 25.0689030, -61.7355347, 61.7641068
16: -45.7322197, 31.5905495, -45.7458572, 31.6296139, -77.3618317, 77.3364105
17: -65.8319016, 41.9321861, -65.6003647, 41.8422699, -107.6741714, 107.5325470
18: -37.7391090, 33.4282646, -37.7378731, 33.6593475, -71.3984528, 71.1661377
19: -31.8081799, 20.4610310, -31.7267780, 20.4778461, -52.2860260, 52.1878052
20: -27.2644730, 20.7651634, -27.1931744, 20.8035412, -48.0680161, 47.9583359
21: -38.3107109, 23.7044353, -38.1834106, 23.6798706, -61.9905777, 61.8878441
22: -37.4188995, 25.0150623, -37.3755226, 24.9752884, -62.3941879, 62.3905869
23: -31.4589901, 22.4349995, -31.4029312, 22.4819107, -53.9408951, 53.8379250
24: -32.4433098, 23.6035194, -32.4534569, 23.6761646, -56.1194763, 56.0569763
25: -29.8187218, 28.2066422, -29.8175011, 28.2530651, -58.0717850, 58.0241394
26: -46.6662674, 33.5693359, -46.4781151, 33.6383972, -80.3046570, 80.0474548
27: -38.5036087, 22.3502789, -38.4381790, 22.3888683, -60.8924789, 60.7884598
28: -31.6911373, 25.9241657, -31.6110897, 25.9852867, -57.6764221, 57.5352516
29: -38.9315720, 24.8247833, -38.8562126, 24.7898140, -63.7213860, 63.6809959
30: -37.5843506, 26.3124886, -37.5268784, 26.3985004, -63.9828491, 63.8393631
31: -36.0240860, 25.5217323, -36.0057373, 25.5912781, -61.6153641, 61.5274696
32: -33.1587982, 25.1883411, -33.1783371, 25.2371826, -58.3959808, 58.3666763
33: -54.7438545, 33.3146057, -54.7302513, 33.2895737, -88.0334320, 88.0448608
34: -47.3275452, 23.5824718, -47.3218842, 23.6433353, -70.9708786, 70.9043579
35: -43.2391472, 28.0445671, -43.2436867, 28.0884743, -71.3276138, 71.2882538
36: -41.4110794, 31.2012653, -41.4156418, 31.2134094, -72.6244812, 72.6169052
37: -60.7519684, 28.0964298, -60.8295059, 28.2973633, -89.0493317, 88.9259338
38: -53.0894699, 40.6387062, -53.0951080, 40.7007141, -93.7901764, 93.7338104
39: -62.1196861, 34.8530350, -62.1607628, 34.7688828, -96.8885651, 97.0137939
40: -54.7274361, 23.3043442, -54.7876701, 23.5137215, -78.2411575, 78.0920105
41: -35.4232101, 21.6561165, -35.4464951, 21.7183704, -57.1415710, 57.1026115
42: -29.7222710, 19.9938335, -29.7086315, 20.0386105, -49.7608719, 49.7024612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=479, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3416742, upper bound: 29.2412320
time: 69.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3435328, upper bound: 29.2752696
time: 111.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -46.2381554, 30.1991024, -46.4409828, 30.2403507, -76.4785080, 76.6400833
1: -27.6837234, 28.5897484, -27.8547649, 28.6621819, -56.3459053, 56.4445114
2: -24.5481014, 25.4131737, -24.7291451, 25.4640408, -50.0121307, 50.1423187
3: -30.6890278, 28.4460735, -30.8890553, 28.5271778, -59.2162018, 59.3351288
4: -29.1190891, 27.0706558, -29.3268242, 27.1409302, -56.2600174, 56.3974800
5: -30.2407913, 30.9517555, -30.4241962, 31.0009098, -61.2416992, 61.3759537
6: -36.2628937, 22.7411556, -36.3420563, 22.9255295, -59.1884155, 59.0832100
7: -35.5731773, 31.8327141, -35.7444992, 31.9203014, -67.4934692, 67.5772095
8: -38.4232140, 29.6755085, -38.6861839, 29.8345318, -68.2577438, 68.3616867
9: -31.8394661, 26.9704533, -31.9054585, 26.9892941, -58.8287582, 58.8759117
10: -43.8091850, 37.8343811, -43.8961983, 37.9277039, -81.7368774, 81.7305756
11: -42.4806862, 33.3302345, -42.4815102, 33.4351082, -75.9157867, 75.8117447
12: -37.7081985, 35.7240486, -37.7800064, 35.9226685, -73.6308441, 73.5040588
13: -40.0215149, 43.3945694, -40.1940231, 43.4558868, -83.4774017, 83.5885925
14: -66.0002213, 23.4234867, -66.0857010, 23.4955711, -89.4957886, 89.5091858
15: -36.7450867, 25.0751743, -36.8888779, 25.1404037, -61.8854904, 61.9640503
16: -45.7634811, 31.6414280, -45.8504982, 31.7696285, -77.5331116, 77.4919128
17: -65.8614960, 42.0861206, -65.8509064, 42.1796684, -108.0411530, 107.9370270
18: -37.7604637, 33.4795990, -37.8227577, 33.7816162, -71.5420837, 71.3023529
19: -31.8265572, 20.5149994, -31.8483925, 20.5987816, -52.4253387, 52.3633919
20: -27.2834377, 20.8094063, -27.2947254, 20.9045982, -48.1880341, 48.1041336
21: -38.3305130, 23.7784405, -38.3343163, 23.8462753, -62.1767883, 62.1127548
22: -37.4416809, 25.0836678, -37.4756165, 25.1335983, -62.5752792, 62.5592842
23: -31.4749870, 22.4727135, -31.4941216, 22.5686188, -54.0436058, 53.9668350
24: -32.4625473, 23.6172180, -32.5196838, 23.7146454, -56.1771927, 56.1369019
25: -29.8372917, 28.2371597, -29.8884792, 28.3313446, -58.1686363, 58.1256409
26: -46.6892128, 33.6664772, -46.6707115, 33.8537369, -80.5429382, 80.3371811
27: -38.5247726, 22.3687458, -38.5124016, 22.4369106, -60.9616852, 60.8811455
28: -31.7076149, 25.9605942, -31.7054310, 26.0690804, -57.7766876, 57.6660233
29: -38.9506493, 24.9169102, -38.9884300, 24.9900074, -63.9406586, 63.9053383
30: -37.5999184, 26.3668137, -37.6048126, 26.5251999, -64.1251221, 63.9716225
31: -36.0454750, 25.5634747, -36.1084900, 25.6890888, -61.7345657, 61.6719627
32: -33.1760101, 25.2402611, -33.2657471, 25.3576164, -58.5336266, 58.5060043
33: -54.8143463, 33.3344269, -54.9039192, 33.3902206, -88.2045670, 88.2383423
34: -47.3923759, 23.5992165, -47.4744720, 23.7493420, -71.1417084, 71.0736847
35: -43.2881241, 28.0557365, -43.3643494, 28.1410599, -71.4291840, 71.4200821
36: -41.4296036, 31.2425919, -41.4847336, 31.3151360, -72.7447357, 72.7273254
37: -60.7811203, 28.1394196, -60.9308052, 28.4073200, -89.1884308, 89.0702209
38: -53.1302910, 40.6657639, -53.2201157, 40.7837791, -93.9140701, 93.8858795
39: -62.1675873, 34.8631821, -62.2885895, 34.8274002, -96.9949875, 97.1517715
40: -54.7770462, 23.3185024, -54.9165688, 23.5720367, -78.3490753, 78.2350693
41: -35.4449768, 21.6908150, -35.5167847, 21.8055344, -57.2505035, 57.2075996
42: -29.7433052, 20.0528507, -29.7848206, 20.1736946, -49.9169922, 49.8376694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=479, inp2_unstable=479, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3423288, upper bound: 29.2789435
time: 69.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2717088, upper bound: 29.3130381
time: 72.52 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -46.3317184, 30.2851009, -46.3150139, 30.1777840, -76.5095062, 76.6001129
1: -27.7456074, 28.7091846, -27.7628918, 28.6717148, -56.4173203, 56.4720726
2: -24.5914116, 25.5407162, -24.6014538, 25.4376316, -50.0290375, 50.1421661
3: -30.7269077, 28.6099129, -30.7246666, 28.4832001, -59.2101059, 59.3345757
4: -29.1559830, 27.2037315, -29.1721668, 27.0953369, -56.2513199, 56.3759003
5: -30.2789688, 31.0715790, -30.2820568, 30.9258423, -61.2048111, 61.3536339
6: -36.3867340, 22.7969837, -36.3350677, 22.8241997, -59.2109299, 59.1320496
7: -35.6315460, 31.9668293, -35.6273956, 31.9302216, -67.5617676, 67.5942230
8: -38.4748459, 29.9073715, -38.5209808, 29.8439445, -68.3187866, 68.4283524
9: -31.8478012, 27.0082512, -31.8159370, 26.9226227, -58.7704239, 58.8241882
10: -43.8792000, 37.8520622, -43.8360710, 37.7614670, -81.6406708, 81.6881332
11: -42.5789452, 33.2694931, -42.3487167, 33.2050972, -75.7840424, 75.6182022
12: -37.8431625, 35.7035599, -37.6208725, 35.6406784, -73.4838257, 73.3244324
13: -40.1013222, 43.5014153, -40.1538467, 43.4274330, -83.5287323, 83.6552582
14: -66.0771866, 23.4284401, -65.9559860, 23.3570175, -89.4342041, 89.3844299
15: -36.7834702, 25.1658020, -36.7241516, 25.1131439, -61.8966064, 61.8899498
16: -45.8691521, 31.6730289, -45.7992401, 31.6419640, -77.5111084, 77.4722672
17: -65.9561768, 42.0152588, -65.6377945, 41.8687057, -107.8248825, 107.6530533
18: -37.9524727, 33.5858383, -37.8331299, 33.6700058, -71.6224823, 71.4189682
19: -31.9107151, 20.5063305, -31.7691231, 20.4834099, -52.3941193, 52.2754517
20: -27.3285351, 20.8341045, -27.2188225, 20.8230991, -48.1516342, 48.0529251
21: -38.4091949, 23.7432442, -38.2206879, 23.6905632, -62.0997581, 61.9639320
22: -37.5318489, 25.0520668, -37.4061508, 24.9879875, -62.5198364, 62.4582176
23: -31.5462589, 22.4821014, -31.4388256, 22.4943409, -54.0405960, 53.9209290
24: -32.5818596, 23.6483917, -32.5079346, 23.6821213, -56.2639771, 56.1563263
25: -29.9099922, 28.2702656, -29.8519821, 28.2640495, -58.1740417, 58.1222458
26: -46.7876740, 33.6553116, -46.5302315, 33.6566582, -80.4443359, 80.1855392
27: -38.5874596, 22.3973866, -38.4654846, 22.4040241, -60.9914856, 60.8628693
28: -31.7692184, 25.9830494, -31.6443062, 26.0007381, -57.7699509, 57.6273575
29: -39.0625305, 24.8823204, -38.8906288, 24.8138161, -63.8763466, 63.7729454
30: -37.6764526, 26.3948784, -37.5600662, 26.4175758, -64.0940247, 63.9549446
31: -36.2117195, 25.6033058, -36.0865936, 25.5988312, -61.8105507, 61.6898994
32: -33.2385712, 25.2643700, -33.2046661, 25.2523727, -58.4909363, 58.4690361
33: -54.8651772, 33.3969269, -54.7823906, 33.3011551, -88.1663208, 88.1793213
34: -47.4520874, 23.6667213, -47.3774185, 23.6555653, -71.1076508, 71.0441360
35: -43.3490295, 28.1085243, -43.2921791, 28.0996475, -71.4486771, 71.4006958
36: -41.4812241, 31.2399349, -41.4439392, 31.2215919, -72.7028198, 72.6838684
37: -60.9923286, 28.2305565, -60.9338722, 28.3065224, -89.2988510, 89.1644287
38: -53.2209091, 40.7302017, -53.1489868, 40.7159729, -93.9368820, 93.8791885
39: -62.2581635, 34.9064941, -62.2142601, 34.7781754, -97.0363312, 97.1207581
40: -54.9452629, 23.4599895, -54.8809662, 23.5274830, -78.4727478, 78.3409576
41: -35.5414658, 21.7098999, -35.4989357, 21.7299805, -57.2714462, 57.2088356
42: -29.7908306, 20.0622311, -29.7378635, 20.0541821, -49.8450089, 49.8000946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=479, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1609

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3416742, upper bound: 29.2727024
time: 121.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3435328, upper bound: 29.3064950
time: 78.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -46.4020157, 30.2982979, -46.4827766, 30.2852783, -76.6872864, 76.7810745
1: -27.7898464, 28.7197037, -27.8698425, 28.7230072, -56.5128479, 56.5895424
2: -24.6531734, 25.5505276, -24.7411804, 25.5279255, -50.1810989, 50.2917099
3: -30.8035564, 28.6242352, -30.8972397, 28.6103077, -59.4138641, 59.5214691
4: -29.2306690, 27.2157726, -29.3378143, 27.2067184, -56.4373856, 56.5535851
5: -30.3455334, 31.0850983, -30.4343433, 31.0613728, -61.4069061, 61.5194397
6: -36.4055176, 22.8453045, -36.4058990, 22.9394836, -59.3449936, 59.2512016
7: -35.6810837, 31.9798717, -35.7559738, 31.9889870, -67.6700668, 67.7358475
8: -38.5529251, 29.9209118, -38.6967773, 29.9486427, -68.5015717, 68.6176910
9: -31.8910866, 27.0327911, -31.9193439, 27.0127926, -58.9038773, 58.9521332
10: -43.9080963, 37.9217987, -43.9294662, 37.9445076, -81.8526001, 81.8512650
11: -42.6000938, 33.3807907, -42.5264091, 33.4496078, -76.0496979, 75.9071960
12: -37.8609619, 35.8387070, -37.8492775, 35.9389114, -73.7998734, 73.6879883
13: -40.1184464, 43.5300522, -40.2109413, 43.5143204, -83.6327667, 83.7409897
14: -66.1151886, 23.5035305, -66.1204987, 23.5245667, -89.6397552, 89.6240311
15: -36.8619308, 25.1812172, -36.9086380, 25.1846199, -62.0465508, 62.0898552
16: -45.9003487, 31.7239590, -45.9037704, 31.7819557, -77.6823044, 77.6277237
17: -65.9858398, 42.1692467, -65.8884354, 42.2060432, -108.1918640, 108.0576782
18: -37.9737778, 33.6372223, -37.9180298, 33.7922592, -71.7660141, 71.5552521
19: -31.9290771, 20.5603104, -31.8907776, 20.6043663, -52.5334435, 52.4510841
20: -27.3474770, 20.8783321, -27.3203926, 20.9241638, -48.2716370, 48.1987228
21: -38.4289856, 23.8172722, -38.3715973, 23.8569679, -62.2859535, 62.1888657
22: -37.5547180, 25.1206245, -37.5062561, 25.1462841, -62.7010040, 62.6268692
23: -31.5622368, 22.5198383, -31.5300102, 22.5810509, -54.1432877, 54.0498466
24: -32.6010742, 23.6621056, -32.5741348, 23.7206211, -56.3216934, 56.2362404
25: -29.9285660, 28.3007832, -29.9229355, 28.3423195, -58.2708855, 58.2237167
26: -46.8106499, 33.7524834, -46.7228088, 33.8719978, -80.6826477, 80.4752884
27: -38.6086502, 22.4158421, -38.5397072, 22.4520588, -61.0607071, 60.9555511
28: -31.7856998, 26.0194664, -31.7386456, 26.0845242, -57.8702164, 57.7581100
29: -39.0815887, 24.9743500, -39.0228539, 25.0139561, -64.0955429, 63.9972000
30: -37.6920166, 26.4492283, -37.6380157, 26.5443020, -64.2363129, 64.0872421
31: -36.2331238, 25.6450329, -36.1893768, 25.6966095, -61.9297333, 61.8344078
32: -33.2558212, 25.3162689, -33.2920761, 25.3727512, -58.6285706, 58.6083450
33: -54.9356842, 33.4167328, -54.9560776, 33.4018669, -88.3375549, 88.3728104
34: -47.5168915, 23.6835003, -47.5299950, 23.7616024, -71.2784958, 71.2134933
35: -43.3980446, 28.1197548, -43.4127808, 28.1522369, -71.5502777, 71.5325317
36: -41.4997330, 31.2812862, -41.5130386, 31.3232841, -72.8230133, 72.7943192
37: -61.0213509, 28.2735786, -61.0350571, 28.4164772, -89.4378281, 89.3086319
38: -53.2617226, 40.7573013, -53.2740059, 40.7990532, -94.0607758, 94.0313110
39: -62.3060760, 34.9166222, -62.3421059, 34.8367233, -97.1427917, 97.2587280
40: -54.9948235, 23.4741211, -55.0098839, 23.5858288, -78.5806427, 78.4840088
41: -35.5632401, 21.7445793, -35.5692062, 21.8170929, -57.3803329, 57.3137856
42: -29.8118534, 20.1212807, -29.8140507, 20.1892662, -50.0011215, 49.9353333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=479, inp2_unstable=479, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1609

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3423288, upper bound: 29.3103935
time: 58.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3442030, upper bound: 29.3442030
time: 69.05 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 129.73 seconds
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 129.73
Output dim: 2, lower bound: -29.3042600, upper bound: 29.3095815
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 129.73
Output dim: 2, lower bound: -29.3060802, upper bound: 29.3432088
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 129.73
Output dim: 2, lower bound: -29.3102398, upper bound: 29.3091590
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 129.73
Output dim: 2, lower bound: -29.2717088, upper bound: 29.3420240
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 129.73
Output dim: 2, lower bound: -29.3416742, upper bound: 29.2412320
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 129.73
Output dim: 2, lower bound: -29.3435328, upper bound: 29.2752696
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 129.73
Output dim: 2, lower bound: -29.3423288, upper bound: 29.2789435
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 129.73
Output dim: 2, lower bound: -29.2717088, upper bound: 29.3130381
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 129.73
Output dim: 2, lower bound: -29.3416742, upper bound: 29.2727024
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 129.73
Output dim: 2, lower bound: -29.3435328, upper bound: 29.3064950
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 129.73
Output dim: 2, lower bound: -29.3423288, upper bound: 29.3103935
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 129.73
Output dim: 2, lower bound: -29.3442030, upper bound: 29.3442030

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -46.1974068, 30.1462669, -46.4113426, 30.2665710, -76.4639740, 76.5576096
1: -27.6665592, 28.6442471, -27.8264847, 28.7084503, -56.3750076, 56.4707260
2: -24.4934196, 25.4096813, -24.6781693, 25.5136757, -50.0070953, 50.0878487
3: -30.6104031, 28.4536591, -30.8231506, 28.5929375, -59.2033386, 59.2768059
4: -29.0449505, 27.0606194, -29.2611713, 27.1891861, -56.2341385, 56.3217888
5: -30.1768227, 30.8988762, -30.3702908, 31.0445709, -61.2213936, 61.2691612
6: -36.2978935, 22.7443848, -36.3807259, 22.9085369, -59.2064285, 59.1251106
7: -35.5303345, 31.9017181, -35.7058258, 31.9735508, -67.5038834, 67.6075439
8: -38.3787003, 29.8087177, -38.6295662, 29.9314518, -68.3101501, 68.4382782
9: -31.7286854, 26.8982716, -31.8567734, 26.9872208, -58.7159042, 58.7550430
10: -43.7654686, 37.6903152, -43.8869629, 37.8761902, -81.6416626, 81.5772781
11: -42.3053932, 33.0659523, -42.5000839, 33.3211594, -75.6265564, 75.5660324
12: -37.5846214, 35.5449638, -37.8266983, 35.8206329, -73.4052582, 73.3716583
13: -40.0175476, 43.3923340, -40.1814499, 43.4841843, -83.5017319, 83.5737839
14: -65.8361206, 23.2612076, -66.0752258, 23.4253025, -89.2614212, 89.3364334
15: -36.6529121, 25.0803318, -36.8313675, 25.1650352, -61.8179474, 61.9116974
16: -45.7406197, 31.5385170, -45.8610153, 31.7293968, -77.4700089, 77.3995285
17: -65.5701599, 41.7094994, -65.8526764, 42.0169258, -107.5870667, 107.5621796
18: -37.7902145, 33.4241486, -37.8930283, 33.7084045, -71.4986115, 71.3171768
19: -31.7364655, 20.3986855, -31.8684769, 20.5399837, -52.2764511, 52.2671623
20: -27.1830082, 20.7463646, -27.2966785, 20.8722191, -48.0552292, 48.0430412
21: -38.1692886, 23.5989895, -38.3437309, 23.7673721, -61.9366608, 61.9427185
22: -37.3497467, 24.9191990, -37.4797363, 25.0683002, -62.4180450, 62.3989334
23: -31.4043522, 22.3906975, -31.5101566, 22.5292969, -53.9336472, 53.9008522
24: -32.4567642, 23.5708694, -32.5510254, 23.6873055, -56.1440697, 56.1218948
25: -29.8042946, 28.1855640, -29.8986282, 28.2995853, -58.1038666, 58.0841904
26: -46.4853020, 33.4792137, -46.6968307, 33.7595139, -80.2448120, 80.1760406
27: -38.4071732, 22.2829247, -38.5106583, 22.3973217, -60.8044891, 60.7935829
28: -31.6110878, 25.8863583, -31.7174702, 26.0312309, -57.6423111, 57.6038246
29: -38.8236504, 24.7181969, -38.9969482, 24.9064388, -63.7300873, 63.7151451
30: -37.5050278, 26.2479820, -37.6153870, 26.4648113, -63.9698410, 63.8633652
31: -36.0379410, 25.4993629, -36.1624527, 25.6392136, -61.6771545, 61.6618118
32: -33.1392288, 25.2104225, -33.2650757, 25.3381824, -58.4774017, 58.4754982
33: -54.6965523, 33.2277222, -54.8683510, 33.3790970, -88.0756531, 88.0960693
34: -47.3388672, 23.5627899, -47.4651642, 23.7429600, -71.0818253, 71.0279541
35: -43.2309570, 28.0405560, -43.3538055, 28.1397991, -71.3707581, 71.3943558
36: -41.3868675, 31.1798248, -41.4856644, 31.2892170, -72.6760788, 72.6654892
37: -60.8645706, 28.1708755, -60.9962311, 28.3856544, -89.2502289, 89.1671066
38: -53.0576248, 40.6634216, -53.2120895, 40.7784500, -93.8360748, 93.8755112
39: -62.0919266, 34.7486191, -62.2647705, 34.8207474, -96.9126740, 97.0133896
40: -54.8272858, 23.3896484, -54.9575653, 23.5711346, -78.3984222, 78.3472137
41: -35.4546356, 21.6515007, -35.5389442, 21.7913742, -57.2460098, 57.1904373
42: -29.7077198, 19.9704552, -29.7872753, 20.1315727, -49.8392944, 49.7577286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=476, inp2_unstable=479, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.3040618, upper bound: 29.3012073
time: 70.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3040618, upper bound: 29.3411821
time: 86.49 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -46.3678398, 30.2529850, -46.3208351, 30.1931229, -76.5609589, 76.5738220
1: -27.7770119, 28.6674080, -27.7581902, 28.6160412, -56.3930511, 56.4255943
2: -24.6422691, 25.4874611, -24.6303082, 25.4008617, -50.0431290, 50.1177673
3: -30.7952442, 28.5604801, -30.7798615, 28.4766598, -59.2719040, 59.3403397
4: -29.2198544, 27.1471291, -29.2059174, 27.0609016, -56.2807541, 56.3530464
5: -30.3355827, 31.0269566, -30.3207302, 30.9422073, -61.2777863, 61.3476868
6: -36.3484344, 22.8318710, -36.2858696, 22.8213100, -59.1697464, 59.1177406
7: -35.6695480, 31.9275341, -35.6461563, 31.8767834, -67.5463333, 67.5736923
8: -38.5422516, 29.8165436, -38.5259552, 29.7273884, -68.2696381, 68.3424988
9: -31.8720169, 27.0081329, -31.8612385, 26.9539490, -58.8259583, 58.8693695
10: -43.8855057, 37.8954849, -43.8578796, 37.8504105, -81.7359161, 81.7533646
11: -42.5367050, 33.3622017, -42.3828125, 33.3457108, -75.8824158, 75.7450104
12: -37.8021889, 35.8215561, -37.7253036, 35.8031769, -73.6053619, 73.5468521
13: -40.0982933, 43.4734116, -40.0984612, 43.3801346, -83.4784241, 83.5718689
14: -66.0797272, 23.4584656, -65.9676514, 23.4264317, -89.5061493, 89.4261169
15: -36.8408356, 25.1531639, -36.8130684, 25.1131458, -61.9539795, 61.9662323
16: -45.8638229, 31.7035255, -45.8072662, 31.6768913, -77.5407104, 77.5107803
17: -65.9468842, 42.1376610, -65.7734528, 42.1287918, -108.0756683, 107.9111176
18: -37.8851929, 33.6199417, -37.7180519, 33.6132545, -71.4984436, 71.3379974
19: -31.8877716, 20.5520611, -31.7964306, 20.5436382, -52.4314117, 52.3484917
20: -27.3223534, 20.8605003, -27.2596893, 20.8425159, -48.1648712, 48.1201897
21: -38.3837662, 23.8057365, -38.2654953, 23.7990303, -62.1827965, 62.0712280
22: -37.5216637, 25.1099548, -37.4094696, 25.1046410, -62.6262970, 62.5194206
23: -31.5129700, 22.5039597, -31.4195538, 22.4938011, -54.0067711, 53.9235077
24: -32.5484009, 23.6511974, -32.4421692, 23.6439476, -56.1923370, 56.0933685
25: -29.8943119, 28.2885838, -29.8383236, 28.2763767, -58.1706848, 58.1269073
26: -46.7249756, 33.7371902, -46.5338593, 33.7246552, -80.4496307, 80.2710419
27: -38.5687561, 22.4039135, -38.4383163, 22.3987179, -60.9674759, 60.8422318
28: -31.7369881, 26.0041256, -31.6339588, 25.9912338, -57.7282181, 57.6380768
29: -39.0407982, 24.9632988, -38.9112587, 24.9624920, -64.0032806, 63.8745575
30: -37.6410446, 26.4296341, -37.5118942, 26.4143734, -64.0554199, 63.9415283
31: -36.1764145, 25.6306496, -36.0559044, 25.6246300, -61.8010330, 61.6865540
32: -33.2260666, 25.2989922, -33.2139473, 25.2796288, -58.5056915, 58.5129356
33: -54.8992271, 33.4014740, -54.8673172, 33.3125687, -88.2117920, 88.2687912
34: -47.4582634, 23.6714630, -47.4010353, 23.6557198, -71.1139832, 71.0724945
35: -43.3545609, 28.1101971, -43.3126526, 28.0811214, -71.4356766, 71.4228363
36: -41.4546280, 31.2739983, -41.4150696, 31.2596149, -72.7142410, 72.6890640
37: -60.9173546, 28.2630806, -60.8079910, 28.2527237, -89.1700745, 89.0710754
38: -53.2121010, 40.7440491, -53.1616135, 40.7160110, -93.9281158, 93.9056625
39: -62.2715836, 34.9033890, -62.2505760, 34.7944870, -97.0660706, 97.1539612
40: -54.9166260, 23.4604759, -54.8374138, 23.4293880, -78.3460083, 78.2978897
41: -35.5012474, 21.7305622, -35.4390488, 21.7138577, -57.2151031, 57.1696014
42: -29.7812233, 20.1040459, -29.7494087, 20.0829067, -49.8641281, 49.8534546

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=478, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.3102797, upper bound: 29.3012846
time: 69.90 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3102797, upper bound: 29.3409488
time: 78.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -46.0733566, 30.0133553, -46.2407227, 30.0478764, -76.1212311, 76.2540741
1: -27.6009769, 28.4645672, -27.7367897, 28.5545406, -56.1555176, 56.2013550
2: -24.3997860, 25.2412605, -24.5719242, 25.2943020, -49.6940842, 49.8131866
3: -30.5357800, 28.2993431, -30.6966209, 28.3353233, -58.8711014, 58.9959641
4: -28.9606152, 26.8924103, -29.1439819, 26.9490852, -55.9096985, 56.0363922
5: -30.0980244, 30.7856846, -30.2524948, 30.7906723, -60.8886948, 61.0381775
6: -36.0442276, 22.6085892, -36.1736641, 22.7924004, -58.8366280, 58.7822533
7: -35.4580612, 31.6792755, -35.5990181, 31.7931652, -67.2512207, 67.2782822
8: -38.2766190, 29.5100155, -38.4934731, 29.6561241, -67.9327393, 68.0034866
9: -31.7425041, 26.8907585, -31.7799244, 26.8740158, -58.6165161, 58.6706848
10: -43.7004280, 37.5954285, -43.7759171, 37.6617661, -81.3621979, 81.3713455
11: -42.3790054, 33.1300964, -42.2691803, 33.1510887, -75.5300903, 75.3992767
12: -37.5324821, 35.5031929, -37.4759254, 35.5937653, -73.1262512, 72.9791183
13: -39.9302025, 43.2501869, -40.1094971, 43.3133240, -83.2435150, 83.3596802
14: -65.8212967, 23.1343956, -65.8827515, 23.2219505, -89.0432434, 89.0171509
15: -36.5988159, 24.9440079, -36.6860733, 25.0110359, -61.6098480, 61.6300735
16: -45.6588974, 31.5051765, -45.7156334, 31.5875626, -77.2464600, 77.2208099
17: -65.6831818, 41.6056023, -65.5686798, 41.6814270, -107.3646088, 107.1742859
18: -37.6743317, 33.3625183, -37.7095108, 33.6317787, -71.3061066, 71.0720291
19: -31.7482109, 20.4175320, -31.7005672, 20.4603214, -52.2085342, 52.1180954
20: -27.1921959, 20.7304039, -27.1588974, 20.7885590, -47.9807549, 47.8893013
21: -38.1998749, 23.6503983, -38.1325417, 23.6568298, -61.8567047, 61.7829399
22: -37.3211288, 24.9448166, -37.3326035, 24.9421062, -62.2632370, 62.2774200
23: -31.3915634, 22.3777294, -31.3727036, 22.4578514, -53.8494148, 53.7504349
24: -32.3506012, 23.5534859, -32.4120560, 23.6551399, -56.0057297, 55.9655380
25: -29.7503376, 28.1528893, -29.7860699, 28.2295322, -57.9798660, 57.9389572
26: -46.5840454, 33.5242958, -46.4420471, 33.6207504, -80.2047958, 79.9663391
27: -38.3326797, 22.3057671, -38.3555679, 22.3752899, -60.7079697, 60.6613350
28: -31.5996284, 25.8825054, -31.5676270, 25.9685745, -57.5681915, 57.4501343
29: -38.8074455, 24.7574043, -38.8024330, 24.7589340, -63.5663757, 63.5598335
30: -37.4597321, 26.2412071, -37.4667015, 26.3718262, -63.8315582, 63.7079086
31: -35.9456139, 25.4647408, -35.9706268, 25.5664444, -61.5120583, 61.4353676
32: -32.9504166, 25.1243782, -33.0777130, 25.2207279, -58.1711426, 58.2020912
33: -54.6195297, 33.2494507, -54.6691208, 33.2693100, -87.8888397, 87.9185638
34: -47.1844673, 23.5308857, -47.2523575, 23.6302090, -70.8146744, 70.7832413
35: -43.1343918, 28.0094223, -43.1927414, 28.0792503, -71.2136383, 71.2021637
36: -41.2233391, 31.1581936, -41.3249817, 31.2038555, -72.4271927, 72.4831696
37: -60.5092278, 27.9915199, -60.7118111, 28.2742386, -88.7834625, 88.7033310
38: -52.9184837, 40.5786667, -53.0140305, 40.6855164, -93.6039886, 93.5926971
39: -61.9254456, 34.7894821, -62.0681496, 34.7517853, -96.6772308, 96.8576279
40: -54.4863472, 23.2302952, -54.6697273, 23.5002594, -77.9865952, 77.9000168
41: -35.2400551, 21.5747318, -35.3575745, 21.7018700, -56.9419250, 56.9323044
42: -29.5739212, 19.9056492, -29.6371365, 20.0183945, -49.5923157, 49.5427818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=478, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3396655, upper bound: 29.1991640
time: 72.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3396655, upper bound: 29.2392430
time: 86.15 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -46.1630669, 30.1768913, -46.2711105, 30.1291027, -76.2921600, 76.4479980
1: -27.6378498, 28.5723724, -27.7471256, 28.6079769, -56.2458191, 56.3194962
2: -24.4842167, 25.3959732, -24.5884991, 25.3706532, -49.8548660, 49.9844742
3: -30.6102276, 28.4250546, -30.7155552, 28.3972645, -59.0074921, 59.1406097
4: -29.0421066, 27.0507622, -29.1601658, 27.0262718, -56.0683746, 56.2109222
5: -30.1717167, 30.9309540, -30.2708168, 30.8623276, -61.0340424, 61.2017708
6: -36.2353973, 22.6903934, -36.2671280, 22.8091450, -59.0445404, 58.9575195
7: -35.5218964, 31.8129349, -35.6150970, 31.8587456, -67.3806305, 67.4280243
8: -38.3432426, 29.6542244, -38.5095100, 29.7265224, -68.0697632, 68.1637344
9: -31.7878494, 26.9410057, -31.7985649, 26.8969517, -58.6847916, 58.7395706
10: -43.7771721, 37.7546616, -43.8014374, 37.7402344, -81.5174103, 81.5560989
11: -42.4519272, 33.2139053, -42.3005638, 33.1883621, -75.6402893, 75.5144653
12: -37.6827850, 35.5848846, -37.5484085, 35.6227036, -73.3054810, 73.1332932
13: -39.9996567, 43.3583755, -40.1348343, 43.3655930, -83.3652496, 83.4932098
14: -65.9562073, 23.3385429, -65.9184723, 23.3238850, -89.2800903, 89.2570038
15: -36.6635017, 25.0542278, -36.7029305, 25.0665245, -61.7300262, 61.7571564
16: -45.7243462, 31.5836296, -45.7425270, 31.6266136, -77.3509598, 77.3261566
17: -65.8265305, 41.9172897, -65.5979919, 41.8359985, -107.6625214, 107.5152664
18: -37.7340889, 33.4198608, -37.7356758, 33.6557922, -71.3898773, 71.1555328
19: -31.8031902, 20.4578857, -31.7246456, 20.4764481, -52.2796402, 52.1825256
20: -27.2582150, 20.7629738, -27.1904984, 20.8025627, -48.0607758, 47.9534721
21: -38.3027458, 23.7015114, -38.1798248, 23.6785564, -61.9813004, 61.8813362
22: -37.4130859, 25.0124340, -37.3730392, 24.9741440, -62.3872147, 62.3854713
23: -31.4528427, 22.4308624, -31.4003296, 22.4800911, -53.9329338, 53.8311882
24: -32.4364929, 23.5988560, -32.4505348, 23.6741905, -56.1106796, 56.0493927
25: -29.8115730, 28.2036514, -29.8144703, 28.2517204, -58.0632935, 58.0181198
26: -46.6585541, 33.5669022, -46.4748344, 33.6373291, -80.2958832, 80.0417328
27: -38.4946671, 22.3479195, -38.4343567, 22.3878498, -60.8825150, 60.7822762
28: -31.6841850, 25.9216347, -31.6080399, 25.9841785, -57.6683655, 57.5296745
29: -38.9220352, 24.8206348, -38.8521881, 24.7880096, -63.7100372, 63.6728210
30: -37.5772133, 26.3085232, -37.5238571, 26.3967705, -63.9739838, 63.8323784
31: -36.0172348, 25.5142899, -36.0028191, 25.5880165, -61.6052513, 61.5171089
32: -33.1486359, 25.1853828, -33.1740150, 25.2359123, -58.3845482, 58.3593979
33: -54.7374001, 33.3118553, -54.7274475, 33.2882729, -88.0256729, 88.0393066
34: -47.3200645, 23.5804062, -47.3186264, 23.6424255, -70.9624863, 70.8990326
35: -43.2319870, 28.0434647, -43.2406082, 28.0879822, -71.3199615, 71.2840729
36: -41.4012146, 31.1998463, -41.4113541, 31.2127781, -72.6139832, 72.6111984
37: -60.7390327, 28.0935745, -60.8240967, 28.2960701, -89.0351028, 88.9176712
38: -53.0782433, 40.6364975, -53.0902214, 40.6996918, -93.7779236, 93.7267151
39: -62.1097450, 34.8506126, -62.1563797, 34.7677536, -96.8775024, 97.0069885
40: -54.7156830, 23.3017921, -54.7826920, 23.5125408, -78.2282181, 78.0844879
41: -35.4144516, 21.6530228, -35.4428024, 21.7170010, -57.1314545, 57.0958252
42: -29.7154007, 19.9900875, -29.7057571, 20.0368900, -49.7522888, 49.6958427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=478, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1541

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3415563, upper bound: 29.2333294
time: 63.06 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2698131, upper bound: 29.2720836
time: 73.99 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 139.36 seconds
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 139.36
Output dim: 2, lower bound: -29.3040618, upper bound: 29.3012073
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 139.36
Output dim: 2, lower bound: -29.3040618, upper bound: 29.3411821
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 139.36
Output dim: 2, lower bound: -29.3102797, upper bound: 29.3012846
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 139.36
Output dim: 2, lower bound: -29.3102797, upper bound: 29.3409488
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 139.36
Output dim: 2, lower bound: -29.3396655, upper bound: 29.1991640
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 139.36
Output dim: 2, lower bound: -29.3396655, upper bound: 29.2392430
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 139.36
Output dim: 2, lower bound: -29.3415563, upper bound: 29.2333294
IS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 139.36
Output dim: 2, lower bound: -29.2698131, upper bound: 29.2720836
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 139.36
Output dim: 2, lower bound: -29.3423288, upper bound: 29.2789435
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 139.36
Output dim: 2, lower bound: -29.3416742, upper bound: 29.2727024
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 139.36
Output dim: 2, lower bound: -29.3435328, upper bound: 29.3064950
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 139.36
Output dim: 2, lower bound: -29.3423288, upper bound: 29.3103935
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 139.36
Output dim: 2, lower bound: -29.3442030, upper bound: 29.3442030

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 76.37 + 3533.32 = 3609.69 seconds
