## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 3600 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.35 + 75.02 = 77.37 seconds
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

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3136239, upper bound: 29.3498756
time: 69.74 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3136239, upper bound: 29.3503749
time: 85.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 155.20 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 155.20
Output dim: 2, lower bound: -29.3136239, upper bound: 29.3498756
IS_A2, status: Status.UNKNOWN, split count: 1, time: 155.20
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

Time for backsubstitution: 1.84 seconds

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
time: 69.39 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3126478, upper bound: 29.3488882
time: 68.85 seconds

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

Time for backsubstitution: 1.87 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1753

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3176166, upper bound: 29.3483383
time: 86.45 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3493929, upper bound: 29.3493934
time: 66.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 154.57 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 154.57
Output dim: 2, lower bound: -29.2807780, upper bound: 29.3478032
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 154.57
Output dim: 2, lower bound: -29.3126478, upper bound: 29.3488882
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 154.57
Output dim: 2, lower bound: -29.3176166, upper bound: 29.3483383
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 154.57
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

Time for backsubstitution: 1.83 seconds

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
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3150392
time: 88.50 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3462751
time: 67.30 seconds

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

Time for backsubstitution: 1.83 seconds

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
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3163312
time: 85.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3112242, upper bound: 29.3474818
time: 75.28 seconds

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

Time for backsubstitution: 1.84 seconds

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
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3161032, upper bound: 29.3155712
time: 72.68 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3462751
time: 74.68 seconds

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

Time for backsubstitution: 1.83 seconds

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
time: 233.94 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3480191, upper bound: 29.3480195
time: 68.51 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 304.39 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 304.39
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3150392
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 304.39
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3462751
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 304.39
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3163312
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 304.39
Output dim: 2, lower bound: -29.3112242, upper bound: 29.3474818
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 304.39
Output dim: 2, lower bound: -29.3161032, upper bound: 29.3155712
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 304.39
Output dim: 2, lower bound: -29.2792537, upper bound: 29.3462751
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 304.39
Output dim: 2, lower bound: -29.3480191, upper bound: 29.3168572
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 304.39
Output dim: 2, lower bound: -29.3480191, upper bound: 29.3480195

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -46.0184441, 30.0223198, -46.2059822, 30.0610409, -76.0794830, 76.2283020
1: -27.5566521, 28.4778957, -27.6941643, 28.5204659, -56.0771103, 56.1720581
2: -24.3897247, 25.2261829, -24.5587635, 25.2626801, -49.6524048, 49.7849464
3: -30.5017643, 28.2283669, -30.7025795, 28.2899475, -58.7917099, 58.9309464
4: -28.9369240, 26.8654594, -29.1255455, 26.9028301, -55.8397522, 55.9910049
5: -30.0761261, 30.7250023, -30.2497311, 30.7690201, -60.8451385, 60.9747314
6: -36.1201820, 22.6454563, -36.1810493, 22.7846069, -58.9047852, 58.8265076
7: -35.4239731, 31.7187061, -35.5767097, 31.7709179, -67.1948929, 67.2954178
8: -38.2517929, 29.4781609, -38.4517136, 29.5325031, -67.7842941, 67.9298706
9: -31.6805801, 26.8286591, -31.7877445, 26.8840618, -58.5646362, 58.6164017
10: -43.6575813, 37.6069679, -43.7635078, 37.7528687, -81.4104462, 81.3704758
11: -42.1422119, 33.0206795, -42.1922646, 33.2153816, -75.3575897, 75.2129440
12: -37.3928299, 35.4376106, -37.4565620, 35.6820526, -73.0748825, 72.8941727
13: -39.9174805, 43.2280273, -40.0477867, 43.2945251, -83.2120056, 83.2758102
14: -65.7049255, 23.1652126, -65.8193512, 23.3090725, -89.0139923, 88.9845581
15: -36.5356064, 24.9602509, -36.7212029, 25.0197506, -61.5553589, 61.6814499
16: -45.5973549, 31.4695988, -45.7028122, 31.6207180, -77.2180710, 77.1724014
17: -65.4241638, 41.6429977, -65.5279770, 41.9375839, -107.3617249, 107.1709747
18: -37.5046768, 33.2740059, -37.5740395, 33.5232391, -71.0279083, 70.8480453
19: -31.6061535, 20.3577976, -31.6583633, 20.4778652, -52.0840149, 52.0161552
20: -27.1103802, 20.6696548, -27.1604309, 20.7733173, -47.8836975, 47.8300858
21: -38.0456467, 23.5632687, -38.1049271, 23.7030182, -61.7486572, 61.6681938
22: -37.2208710, 24.8837242, -37.3151512, 25.0085030, -62.2293701, 62.1988716
23: -31.2830982, 22.3411198, -31.3159618, 22.4331398, -53.7162323, 53.6570816
24: -32.2855301, 23.5273705, -32.3558502, 23.6034966, -55.8890266, 55.8832169
25: -29.6975498, 28.1223545, -29.7609825, 28.2197800, -57.9173279, 57.8833389
26: -46.2995186, 33.3954315, -46.3270912, 33.6033821, -79.9028931, 79.7225189
27: -38.3062019, 22.2353630, -38.3685532, 22.3323669, -60.6385651, 60.6039162
28: -31.5012150, 25.8218231, -31.5306129, 25.9240074, -57.4252167, 57.3524361
29: -38.6771317, 24.6687469, -38.7773666, 24.8417435, -63.5188599, 63.4461098
30: -37.3802567, 26.1616440, -37.4238205, 26.3174458, -63.6977005, 63.5854645
31: -35.8116531, 25.4236679, -35.9030151, 25.5634651, -61.3751068, 61.3266792
32: -33.0544777, 25.1309795, -33.1250114, 25.2333202, -58.2877960, 58.2559891
33: -54.5627098, 33.1404419, -54.7240257, 33.2296524, -87.7923584, 87.8644714
34: -47.1790962, 23.4742832, -47.2831192, 23.5618896, -70.7409821, 70.7574005
35: -43.1002426, 27.9725189, -43.2063332, 28.0350723, -71.1353149, 71.1788483
36: -41.2953720, 31.1427727, -41.3436546, 31.2125683, -72.5079346, 72.4864197
37: -60.5534172, 28.0383148, -60.6469460, 28.2053642, -88.7587738, 88.6852570
38: -52.9100037, 40.5693054, -53.0340424, 40.6661758, -93.5761719, 93.6033478
39: -61.9476967, 34.6957397, -62.1147575, 34.7522507, -96.6999512, 96.8104935
40: -54.5648232, 23.2295361, -54.6919250, 23.3811188, -77.9459381, 77.9214630
41: -35.2960892, 21.6017113, -35.3454704, 21.6840134, -56.9800873, 56.9471741
42: -29.6258049, 19.9012299, -29.6722870, 20.0146561, -49.6404572, 49.5735130

Time for backsubstitution: 1.73 seconds

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
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1789
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
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1644
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
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1638
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
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1571
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
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1537
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
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1771

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2750173, upper bound: 29.2745930
time: 80.33 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3122347
time: 88.91 seconds

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

Time for backsubstitution: 1.84 seconds

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

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3058565
time: 82.66 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3058565
time: 70.79 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -46.0480537, 30.0585117, -46.3660698, 30.1493607, -76.1974106, 76.4245834
1: -27.5679646, 28.5232811, -27.8052197, 28.6245327, -56.1924973, 56.3284988
2: -24.3985825, 25.2817764, -24.6687908, 25.3866386, -49.7852173, 49.9505653
3: -30.5080051, 28.2853661, -30.8190060, 28.4207211, -58.9287109, 59.1043625
4: -28.9454899, 26.9262085, -29.2564487, 27.0453033, -55.9907913, 56.1826553
5: -30.0835495, 30.7758789, -30.3622704, 30.8851547, -60.9687042, 61.1381454
6: -36.1684189, 22.6564407, -36.2967873, 22.9017544, -59.0701675, 58.9532280
7: -35.4338074, 31.7642040, -35.6858559, 31.8803005, -67.3141098, 67.4500580
8: -38.2605896, 29.5746708, -38.6217041, 29.7503281, -68.0109177, 68.1963730
9: -31.6913261, 26.8482170, -31.8424492, 26.9405689, -58.6318970, 58.6906662
10: -43.6770325, 37.6233292, -43.8337822, 37.8426208, -81.5196533, 81.4571075
11: -42.1979370, 33.0342484, -42.3322754, 33.3175735, -75.5155029, 75.3665237
12: -37.4440002, 35.4508667, -37.5772934, 35.8161201, -73.2601166, 73.0281601
13: -39.9328995, 43.2767029, -40.1583252, 43.4251175, -83.3580170, 83.4350204
14: -65.7342606, 23.2001896, -65.9695969, 23.4026222, -89.1368713, 89.1697845
15: -36.5540085, 24.9827423, -36.8153687, 25.0887985, -61.6428070, 61.7981033
16: -45.6257820, 31.4832344, -45.7958069, 31.7230206, -77.3488007, 77.2790375
17: -65.4578094, 41.6593323, -65.6404114, 42.0085831, -107.4663925, 107.2997360
18: -37.5880814, 33.2829475, -37.7716866, 33.6987991, -71.2868805, 71.0546341
19: -31.6423779, 20.3629341, -31.7504902, 20.5372124, -52.1795883, 52.1134224
20: -27.1290951, 20.6854687, -27.2183456, 20.8541756, -47.9832687, 47.9038124
21: -38.0828171, 23.5718689, -38.2073059, 23.7596283, -61.8424377, 61.7791748
22: -37.2486725, 24.8917160, -37.4096069, 25.0489635, -62.2976303, 62.3013153
23: -31.3262005, 22.3529091, -31.4237595, 22.5187149, -53.8449097, 53.7766647
24: -32.3312454, 23.5336800, -32.4848022, 23.6782093, -56.0094452, 56.0184784
25: -29.7245941, 28.1316128, -29.8425636, 28.2844009, -58.0089951, 57.9741745
26: -46.3773689, 33.4083099, -46.5127182, 33.7497559, -80.1271210, 79.9210281
27: -38.3370895, 22.2447853, -38.4660492, 22.3846912, -60.7217789, 60.7108345
28: -31.5429020, 25.8347912, -31.6322155, 26.0164528, -57.5593491, 57.4670067
29: -38.7084084, 24.6755505, -38.8847923, 24.8913727, -63.5997772, 63.5603409
30: -37.4240341, 26.1772938, -37.5467987, 26.4457588, -63.8697929, 63.7240906
31: -35.8612518, 25.4306526, -36.0332489, 25.6322536, -61.4934998, 61.4638939
32: -33.0739136, 25.1453762, -33.1987915, 25.3253212, -58.3992348, 58.3441658
33: -54.5925980, 33.1529503, -54.8099747, 33.3176842, -87.9102783, 87.9629211
34: -47.2301369, 23.4843025, -47.4087410, 23.6669235, -70.8970642, 70.8930435
35: -43.1365356, 27.9809952, -43.3032761, 28.1057415, -71.2422791, 71.2842636
36: -41.3306198, 31.1487064, -41.4373055, 31.2756748, -72.6062927, 72.5860138
37: -60.6444130, 28.0459938, -60.8684731, 28.3678474, -89.0122528, 88.9144592
38: -52.9482841, 40.5803604, -53.1414185, 40.7483978, -93.6966705, 93.7217789
39: -61.9720688, 34.7065659, -62.2018547, 34.7934418, -96.7655106, 96.9084167
40: -54.6312027, 23.2406940, -54.8593330, 23.5364437, -78.1676483, 78.1000290
41: -35.3491440, 21.6126785, -35.4717140, 21.7859879, -57.1351318, 57.0843925
42: -29.6494484, 19.9148064, -29.7337990, 20.1194496, -49.7688980, 49.6486053

Time for backsubstitution: 1.84 seconds

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
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1781
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

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3069861, upper bound: 29.2757551
time: 80.44 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3074997, upper bound: 29.3134814
time: 80.75 seconds

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

Time for backsubstitution: 1.75 seconds

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
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3069861, upper bound: 29.3069979
time: 83.96 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3074997, upper bound: 29.3446372
time: 79.93 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -46.2176895, 30.1649647, -46.2747040, 30.0757523, -76.2934341, 76.4396667
1: -27.6779785, 28.5462151, -27.7363224, 28.5317249, -56.2097015, 56.2825394
2: -24.5468693, 25.3594589, -24.6201782, 25.2736816, -49.8205490, 49.9796371
3: -30.6922607, 28.3919621, -30.7750530, 28.3041801, -58.9964371, 59.1670151
4: -29.1196651, 27.0124893, -29.2001972, 26.9168415, -56.0365067, 56.2126846
5: -30.2419014, 30.9037247, -30.3121948, 30.7825813, -61.0244827, 61.2159195
6: -36.2187500, 22.7412872, -36.2018585, 22.8115978, -59.0303421, 58.9431458
7: -35.5721855, 31.7896309, -35.6255264, 31.7831631, -67.3553467, 67.4151535
8: -38.4235001, 29.5822887, -38.5173531, 29.5461407, -67.9696426, 68.0996399
9: -31.8341217, 26.9578686, -31.8462582, 26.9072990, -58.7414207, 58.8041267
10: -43.7962074, 37.8279190, -43.8035927, 37.8160782, -81.6122742, 81.6315155
11: -42.4288788, 33.3293800, -42.2150497, 33.3402176, -75.7690964, 75.5444260
12: -37.6612015, 35.7265091, -37.4756165, 35.7973518, -73.4585419, 73.2021255
13: -40.0130157, 43.3577919, -40.0745087, 43.3207626, -83.3337784, 83.4322968
14: -65.9770508, 23.3970795, -65.8608704, 23.4033508, -89.3804016, 89.2579498
15: -36.7401543, 25.0555229, -36.7954178, 25.0367680, -61.7769241, 61.8509407
16: -45.7485924, 31.6472588, -45.7417793, 31.6688976, -77.4174881, 77.3890381
17: -65.8338394, 42.0866165, -65.5608673, 42.1189232, -107.9527588, 107.6474838
18: -37.6829109, 33.4781036, -37.5965233, 33.6028595, -71.2857666, 71.0746231
19: -31.7933884, 20.5158081, -31.6782341, 20.5403118, -52.3336945, 52.1940346
20: -27.2680035, 20.7990246, -27.1811085, 20.8237572, -48.0917587, 47.9801292
21: -38.2969398, 23.7779236, -38.1289291, 23.7903786, -62.0873184, 61.9068527
22: -37.4191818, 25.0821667, -37.3385849, 25.0847569, -62.5039368, 62.4207497
23: -31.4344730, 22.4657078, -31.3329659, 22.4825153, -53.9169884, 53.7986755
24: -32.4225769, 23.6136398, -32.3753433, 23.6344681, -56.0570374, 55.9889832
25: -29.8142719, 28.2331829, -29.7816467, 28.2605896, -58.0748596, 58.0148315
26: -46.6163902, 33.6656761, -46.3489799, 33.7138977, -80.3302917, 80.0146484
27: -38.4981995, 22.3648815, -38.3932800, 22.3850269, -60.8832245, 60.7581635
28: -31.6686039, 25.9520454, -31.5485897, 25.9757690, -57.6443710, 57.5006332
29: -38.9247742, 24.9201221, -38.7984848, 24.9465981, -63.8713722, 63.7186050
30: -37.5598373, 26.3581600, -37.4431305, 26.3943787, -63.9542160, 63.8012924
31: -35.9995117, 25.5613079, -35.9266090, 25.6168633, -61.6163712, 61.4879112
32: -33.1605835, 25.2332497, -33.1474228, 25.2658501, -58.4264336, 58.3806725
33: -54.7945938, 33.3261375, -54.8079910, 33.2505569, -88.0451508, 88.1341248
34: -47.3491287, 23.5924129, -47.3441010, 23.5791912, -70.9283218, 70.9365082
35: -43.2595406, 28.0503769, -43.2614746, 28.0467529, -71.3062897, 71.3118439
36: -41.3979836, 31.2425938, -41.3661995, 31.2456360, -72.6436157, 72.6087952
37: -60.6967964, 28.1379128, -60.6798286, 28.2342949, -88.9310913, 88.8177338
38: -53.1019669, 40.6606331, -53.0894241, 40.6854401, -93.7874069, 93.7500610
39: -62.1512871, 34.8610535, -62.1867981, 34.7667351, -96.9180222, 97.0478516
40: -54.7198448, 23.3110256, -54.7387657, 23.3941154, -78.1139603, 78.0497894
41: -35.3956909, 21.6897583, -35.3717422, 21.7057552, -57.1014481, 57.0615005
42: -29.7227936, 20.0475483, -29.6958923, 20.0698185, -49.7926102, 49.7434349

Time for backsubstitution: 1.80 seconds

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
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1636
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
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1644
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
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1638
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
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1653
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
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1571
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
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1781
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2750173, upper bound: 29.2755156
time: 72.93 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3136875, upper bound: 29.3132035
time: 68.87 seconds

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

Time for backsubstitution: 1.86 seconds

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
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3067859
time: 78.60 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3136875, upper bound: 29.3444516
time: 82.43 seconds

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

Time for backsubstitution: 1.95 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3449520, upper bound: 29.2767014
time: 80.77 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3456274, upper bound: 29.3144508
time: 80.94 seconds

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

Time for backsubstitution: 2.00 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1673

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3449520, upper bound: 29.3079366
time: 88.12 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3456274, upper bound: 29.3456277
time: 73.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 163.95 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.2750173, upper bound: 29.2745930
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3122347
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3058565
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3058565
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.3069861, upper bound: 29.2757551
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.3074997, upper bound: 29.3134814
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.3069861, upper bound: 29.3069979
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.3074997, upper bound: 29.3446372
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.2750173, upper bound: 29.2755156
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.3136875, upper bound: 29.3132035
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3067859
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.3136875, upper bound: 29.3444516
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.3449520, upper bound: 29.2767014
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.3456274, upper bound: 29.3144508
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.3449520, upper bound: 29.3079366
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 163.95
Output dim: 2, lower bound: -29.3456274, upper bound: 29.3456277

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -45.9398613, 30.0070419, -46.0451088, 30.0299187, -75.9697723, 76.0521545
1: -27.5074043, 28.4656944, -27.5950737, 28.4955940, -56.0029984, 56.0607643
2: -24.3212738, 25.2144737, -24.4186268, 25.2388287, -49.5600967, 49.6331024
3: -30.4166107, 28.2112370, -30.5283012, 28.2550640, -58.6716614, 58.7395401
4: -28.8542843, 26.8509655, -28.9563274, 26.8732338, -55.7275085, 55.8072891
5: -30.0018997, 30.7087345, -30.0976219, 30.7358360, -60.7377243, 60.8063507
6: -36.0975037, 22.5897846, -36.1348495, 22.6704884, -58.7679901, 58.7246323
7: -35.3674355, 31.7034893, -35.4601517, 31.7399178, -67.1073532, 67.1636429
8: -38.1656113, 29.4614391, -38.2753906, 29.4983463, -67.6639557, 67.7368317
9: -31.6323757, 26.7971802, -31.6891289, 26.8202515, -58.4526215, 58.4863052
10: -43.6238327, 37.5289612, -43.6934700, 37.5931625, -81.2169952, 81.2224274
11: -42.1171646, 32.8981094, -42.1411400, 32.9645691, -75.0817337, 75.0392456
12: -37.3709450, 35.2881775, -37.4118423, 35.3760300, -72.7469635, 72.7000198
13: -39.8941002, 43.1877518, -40.0000687, 43.2123184, -83.1064148, 83.1878204
14: -65.6610260, 23.0821228, -65.7297363, 23.1406517, -88.8016815, 88.8118591
15: -36.4468384, 24.9421825, -36.5380478, 24.9829025, -61.4297333, 61.4802322
16: -45.5545540, 31.4006214, -45.6161346, 31.4799061, -77.0344391, 77.0167542
17: -65.3885193, 41.4736595, -65.4551239, 41.5908775, -106.9794006, 106.9287872
18: -37.4771042, 33.2160912, -37.5179596, 33.4047012, -70.8818054, 70.7340469
19: -31.5848656, 20.2984352, -31.6149235, 20.3563728, -51.9412384, 51.9133568
20: -27.0879822, 20.6204300, -27.1147156, 20.6725445, -47.7605286, 47.7351456
21: -38.0222054, 23.4818592, -38.0571709, 23.5363655, -61.5585632, 61.5390320
22: -37.1928177, 24.8091164, -37.2578468, 24.8589859, -62.0517883, 62.0669632
23: -31.2646713, 22.2991276, -31.2783489, 22.3472404, -53.6119080, 53.5774727
24: -32.2606506, 23.5112724, -32.3053360, 23.5707207, -55.8313713, 55.8166084
25: -29.6751804, 28.0879765, -29.7154217, 28.1496124, -57.8247833, 57.8033981
26: -46.2718620, 33.2870522, -46.2706604, 33.3820801, -79.6539459, 79.5577087
27: -38.2804222, 22.2124729, -38.3160629, 22.2855930, -60.5660172, 60.5285301
28: -31.4821472, 25.7814007, -31.4916935, 25.8413601, -57.3235092, 57.2730904
29: -38.6528397, 24.5674019, -38.7278175, 24.6358185, -63.2886581, 63.2952194
30: -37.3610535, 26.1012440, -37.3847198, 26.1939564, -63.5550079, 63.4859619
31: -35.7866211, 25.3776932, -35.8520279, 25.4694023, -61.2560234, 61.2297173
32: -33.0335770, 25.0722618, -33.0823364, 25.1132126, -58.1467896, 58.1545982
33: -54.4829140, 33.1165237, -54.5609856, 33.1808510, -87.6637650, 87.6775055
34: -47.1069336, 23.4542084, -47.1356506, 23.5209694, -70.6278992, 70.5898590
35: -43.0443954, 27.9583549, -43.0922203, 28.0062122, -71.0505981, 71.0505753
36: -41.2739182, 31.0959110, -41.2997437, 31.1174927, -72.3914032, 72.3956528
37: -60.5187721, 27.9899559, -60.5758514, 28.1063728, -88.6251450, 88.5658112
38: -52.8609810, 40.5365982, -52.9337158, 40.5995445, -93.4605255, 93.4703140
39: -61.8925858, 34.6766052, -62.0023155, 34.7133904, -96.6059723, 96.6789246
40: -54.5069237, 23.2119141, -54.5742607, 23.3457031, -77.8526306, 77.7861786
41: -35.2710495, 21.5579147, -35.2942772, 21.5946827, -56.8657227, 56.8521919
42: -29.6019211, 19.8354759, -29.6236115, 19.8798256, -49.4817429, 49.4590874

Time for backsubstitution: 1.86 seconds

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
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1629
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
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1762
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
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1623
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
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1690
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
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2717088, upper bound: 29.2392394
time: 71.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2717088, upper bound: 29.2731594
time: 76.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -46.0086823, 30.0198708, -46.2115135, 30.1370907, -76.1457748, 76.2313843
1: -27.5507336, 28.4757271, -27.7011585, 28.5464668, -56.0971985, 56.1768837
2: -24.3815517, 25.2240524, -24.5569897, 25.3289452, -49.7104950, 49.7810402
3: -30.4918098, 28.2251282, -30.6994896, 28.3817921, -58.8735962, 58.9246178
4: -28.9271069, 26.8626251, -29.1203251, 26.9841862, -55.9112930, 55.9829483
5: -30.0672283, 30.7218609, -30.2487240, 30.8709469, -60.9381752, 60.9705811
6: -36.1158295, 22.6317158, -36.2052231, 22.7785130, -58.8943329, 58.8369370
7: -35.4143753, 31.7158318, -35.5860596, 31.7983074, -67.2126846, 67.3018875
8: -38.2421112, 29.4745464, -38.4498558, 29.6027126, -67.8448257, 67.9244003
9: -31.6746731, 26.8213291, -31.7916737, 26.9093227, -58.5839920, 58.6130028
10: -43.6502419, 37.5965195, -43.7847977, 37.7741470, -81.4243927, 81.3813171
11: -42.1378937, 33.0068855, -42.3183823, 33.2067719, -75.3446655, 75.3252716
12: -37.3882904, 35.4209366, -37.6398697, 35.6720619, -73.0603485, 73.0608063
13: -39.9099159, 43.2157898, -40.0560532, 43.2985153, -83.2084351, 83.2718430
14: -65.6979446, 23.1559048, -65.8930511, 23.3069077, -89.0048523, 89.0489578
15: -36.5207825, 24.9572411, -36.7188148, 25.0541077, -61.5748901, 61.6760483
16: -45.5830269, 31.4492741, -45.7180977, 31.6178703, -77.2008972, 77.1673660
17: -65.4175720, 41.6248817, -65.7051620, 41.9257507, -107.3433228, 107.3300476
18: -37.4984093, 33.2660217, -37.6023026, 33.5257187, -71.0241241, 70.8683167
19: -31.6027546, 20.3514137, -31.7360878, 20.4764442, -52.0792007, 52.0875015
20: -27.1065617, 20.6637821, -27.2157536, 20.7727776, -47.8793411, 47.8795357
21: -38.0416374, 23.5545120, -38.2076149, 23.7014523, -61.7430878, 61.7621231
22: -37.2147903, 24.8767929, -37.3571854, 25.0163193, -62.2311096, 62.2339706
23: -31.2801647, 22.3358955, -31.3689919, 22.4331017, -53.7132645, 53.7048874
24: -32.2792740, 23.5243587, -32.3705559, 23.6086655, -55.8879395, 55.8949127
25: -29.6930866, 28.1156483, -29.7856503, 28.2253742, -57.9184570, 57.9012985
26: -46.2937355, 33.3827667, -46.4623413, 33.5959435, -79.8896790, 79.8451004
27: -38.3014297, 22.2287254, -38.3896980, 22.3308964, -60.6323242, 60.6184235
28: -31.4982700, 25.8170166, -31.5856380, 25.9244194, -57.4226837, 57.4026566
29: -38.6709862, 24.6580887, -38.8591270, 24.8346462, -63.5056305, 63.5172157
30: -37.3763046, 26.1538906, -37.4622307, 26.3191051, -63.6953964, 63.6161194
31: -35.8076172, 25.4182663, -35.9542542, 25.5661869, -61.3738022, 61.3725204
32: -33.0501709, 25.1229668, -33.1693192, 25.2323017, -58.2824707, 58.2922783
33: -54.5517235, 33.1357117, -54.7331543, 33.2808685, -87.8325958, 87.8688660
34: -47.1707382, 23.4706097, -47.2872467, 23.6265526, -70.7972870, 70.7578506
35: -43.0918922, 27.9692745, -43.2114868, 28.0585556, -71.1504440, 71.1807556
36: -41.2913513, 31.1366501, -41.3679962, 31.2185707, -72.5099182, 72.5046463
37: -60.5460968, 28.0318604, -60.6757393, 28.2153912, -88.7614899, 88.7075958
38: -52.8990898, 40.5631218, -53.0556831, 40.6819687, -93.5810547, 93.6188049
39: -61.9388237, 34.6868439, -62.1285706, 34.7713852, -96.7102051, 96.8154068
40: -54.5548286, 23.2254028, -54.7018356, 23.4032898, -77.9581146, 77.9272308
41: -35.2920418, 21.5898495, -35.3639374, 21.6792355, -56.9712753, 56.9537811
42: -29.6223965, 19.8921871, -29.6993847, 20.0129242, -49.6353226, 49.5915718

Time for backsubstitution: 1.86 seconds

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
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1722
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
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1644
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
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1623
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
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1690
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
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1541

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2721565, upper bound: 29.2768778
time: 80.51 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2740850, upper bound: 29.3108072
time: 125.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -46.1040230, 30.1061859, -46.0871925, 30.0747986, -76.1788177, 76.1933746
1: -27.6136589, 28.5955963, -27.6102638, 28.5563507, -56.1700096, 56.2058601
2: -24.4263859, 25.3518333, -24.4307423, 25.3026791, -49.7290649, 49.7825775
3: -30.5311756, 28.3893681, -30.5364895, 28.3381176, -58.8692932, 58.9258575
4: -28.9659405, 26.9960670, -28.9673481, 26.9389610, -55.9048996, 55.9634094
5: -30.1066532, 30.8421135, -30.1078072, 30.7962475, -60.9028931, 60.9499207
6: -36.2399483, 22.6939373, -36.1985130, 22.6845398, -58.9244843, 58.8924484
7: -35.4754181, 31.8505745, -35.4717140, 31.8085670, -67.2839813, 67.3222885
8: -38.2953339, 29.7066994, -38.2860374, 29.6123447, -67.9076691, 67.9927368
9: -31.6839695, 26.8593216, -31.7031097, 26.8435802, -58.5275421, 58.5624313
10: -43.7227249, 37.6163635, -43.7268143, 37.6100464, -81.3327713, 81.3431778
11: -42.2365379, 32.9486923, -42.1857529, 32.9795685, -75.2161026, 75.1344452
12: -37.5236816, 35.4028702, -37.4811172, 35.3923492, -72.9160233, 72.8839798
13: -39.9910355, 43.3229065, -40.0171127, 43.2706032, -83.2616272, 83.3400192
14: -65.7757568, 23.1618977, -65.7645874, 23.1691532, -88.9449005, 88.9264832
15: -36.5640984, 25.0482693, -36.5579376, 25.0271206, -61.5912018, 61.6062088
16: -45.6914597, 31.4832325, -45.6694183, 31.4924717, -77.1839218, 77.1526489
17: -65.5127411, 41.5564461, -65.4923859, 41.6172523, -107.1299896, 107.0488281
18: -37.6903992, 33.3736725, -37.6130714, 33.4154472, -71.1058502, 70.9867401
19: -31.6873112, 20.3437424, -31.6571903, 20.3619576, -52.0492706, 52.0009308
20: -27.1519260, 20.6895428, -27.1403046, 20.6923065, -47.8442307, 47.8298454
21: -38.1205940, 23.5206604, -38.0943565, 23.5470695, -61.6676636, 61.6150131
22: -37.3062973, 24.8461094, -37.2886734, 24.8716621, -62.1779594, 62.1347809
23: -31.3518257, 22.3462715, -31.3141327, 22.3598671, -53.7116928, 53.6604042
24: -32.3990707, 23.5561600, -32.3597031, 23.5766773, -55.9757462, 55.9158554
25: -29.7664299, 28.1516972, -29.7498474, 28.1606503, -57.9270782, 57.9015427
26: -46.3932114, 33.3730164, -46.3226967, 33.4004898, -79.7937012, 79.6957092
27: -38.3642044, 22.2594528, -38.3432922, 22.3007660, -60.6649628, 60.6027412
28: -31.5601215, 25.8404045, -31.5249004, 25.8570538, -57.4171753, 57.3653030
29: -38.7837715, 24.6248627, -38.7621346, 24.6598415, -63.4436035, 63.3869972
30: -37.4530907, 26.1836567, -37.4178162, 26.2131386, -63.6662292, 63.6014709
31: -35.9739799, 25.4592381, -35.9326019, 25.4770432, -61.4510193, 61.3918343
32: -33.1132660, 25.1483459, -33.1086426, 25.1285629, -58.2418213, 58.2569885
33: -54.6042099, 33.1988754, -54.6130600, 33.1924324, -87.7966461, 87.8119354
34: -47.2314453, 23.5384712, -47.1910667, 23.5332909, -70.7647247, 70.7295380
35: -43.1542435, 28.0222988, -43.1405487, 28.0173817, -71.1716232, 71.1628418
36: -41.3439865, 31.1346149, -41.3279915, 31.1256943, -72.4696732, 72.4626007
37: -60.7590828, 28.1240807, -60.6800575, 28.1155338, -88.8746185, 88.8041382
38: -52.9923096, 40.6281281, -52.9875450, 40.6149521, -93.6072617, 93.6156769
39: -62.0309792, 34.7300949, -62.0558052, 34.7227173, -96.7536926, 96.7858963
40: -54.7247047, 23.3676510, -54.6674652, 23.3594933, -78.0841980, 78.0351181
41: -35.3891449, 21.6116905, -35.3464966, 21.6063499, -56.9954948, 56.9581871
42: -29.6703587, 19.9039307, -29.6526299, 19.8955860, -49.5659447, 49.5565605

Time for backsubstitution: 1.87 seconds

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
type: A, layer: 1, pos: 1673
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
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1713
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
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1623
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
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1690
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

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2717088, upper bound: 29.2707280
time: 66.57 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2735856, upper bound: 29.3044100
time: 89.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -46.1728058, 30.1190491, -46.2536011, 30.1820030, -76.3548126, 76.3726501
1: -27.6569939, 28.6056290, -27.7163563, 28.6072311, -56.2642250, 56.3219795
2: -24.4866619, 25.3614388, -24.5691090, 25.3927803, -49.8794403, 49.9305458
3: -30.6063671, 28.4032383, -30.7076893, 28.4648361, -59.0711975, 59.1109238
4: -29.0387344, 27.0077057, -29.1313477, 27.0499363, -56.0886688, 56.1390495
5: -30.1719627, 30.8552246, -30.2589169, 30.9314289, -61.1033859, 61.1141434
6: -36.2582664, 22.7358761, -36.2689133, 22.7925262, -59.0507927, 59.0047874
7: -35.5223465, 31.8629284, -35.5975723, 31.8669090, -67.3892517, 67.4604950
8: -38.3718758, 29.7198486, -38.4604568, 29.7167053, -68.0885773, 68.1803055
9: -31.7263107, 26.8834705, -31.8056507, 26.9325905, -58.6588898, 58.6891174
10: -43.7491074, 37.6839333, -43.8181267, 37.7910538, -81.5401611, 81.5020599
11: -42.2572899, 33.0574379, -42.3630180, 33.2217712, -75.4790649, 75.4204559
12: -37.5410118, 35.5356979, -37.7090988, 35.6883583, -73.2293701, 73.2447968
13: -40.0068436, 43.3508873, -40.0730782, 43.3567734, -83.3636169, 83.4239655
14: -65.8128052, 23.2357254, -65.9278870, 23.3354816, -89.1482849, 89.1636124
15: -36.6380005, 25.0632858, -36.7386513, 25.0982780, -61.7362671, 61.8019371
16: -45.7198181, 31.5318851, -45.7712440, 31.6303654, -77.3501816, 77.3031311
17: -65.5419083, 41.7076874, -65.7424927, 41.9520988, -107.4940033, 107.4501724
18: -37.7116776, 33.4236946, -37.6974068, 33.5364914, -71.2481537, 71.1211014
19: -31.7051964, 20.3967361, -31.7783813, 20.4820633, -52.1872559, 52.1751137
20: -27.1704826, 20.7329082, -27.2413807, 20.7925224, -47.9630051, 47.9742889
21: -38.1400490, 23.5932884, -38.2448196, 23.7121525, -61.8522034, 61.8381042
22: -37.3283348, 24.9137383, -37.3880157, 25.0289803, -62.3573151, 62.3017540
23: -31.3673325, 22.3830757, -31.4048290, 22.4457092, -53.8130417, 53.7878952
24: -32.4177170, 23.5692501, -32.4249191, 23.6146221, -56.0323257, 55.9941673
25: -29.7843361, 28.1793480, -29.8201027, 28.2363605, -58.0206985, 57.9994431
26: -46.4150848, 33.4687424, -46.5143967, 33.6143799, -80.0294647, 79.9831390
27: -38.3852348, 22.2756805, -38.4169083, 22.3460312, -60.7312622, 60.6925850
28: -31.5762367, 25.8760529, -31.6188641, 25.9401207, -57.5163574, 57.4949150
29: -38.8019524, 24.7155247, -38.8934326, 24.8586082, -63.6605606, 63.6089554
30: -37.4683380, 26.2363129, -37.4953461, 26.3383026, -63.8066406, 63.7316513
31: -35.9949722, 25.4998169, -36.0348511, 25.5737801, -61.5687523, 61.5346680
32: -33.1298447, 25.1990547, -33.1956253, 25.2476425, -58.3774796, 58.3946800
33: -54.6729965, 33.2180710, -54.7851982, 33.2925949, -87.9655914, 88.0032654
34: -47.2951736, 23.5549393, -47.3426628, 23.6389580, -70.9341278, 70.8975983
35: -43.2016907, 28.0332565, -43.2598457, 28.0697556, -71.2714310, 71.2931061
36: -41.3614273, 31.1753578, -41.3962593, 31.2267704, -72.5881958, 72.5716171
37: -60.7863274, 28.1660385, -60.7799263, 28.2245598, -89.0108795, 88.9459686
38: -53.0304184, 40.6546707, -53.1094437, 40.6974144, -93.7278290, 93.7641144
39: -62.0771751, 34.7403183, -62.1820145, 34.7807465, -96.8579102, 96.9223251
40: -54.7726173, 23.3811207, -54.7950096, 23.4171352, -78.1897430, 78.1761246
41: -35.4101410, 21.6436443, -35.4161606, 21.6908817, -57.1010208, 57.0597992
42: -29.6908360, 19.9606800, -29.7284069, 20.0286465, -49.7194824, 49.6890831

Time for backsubstitution: 1.89 seconds

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
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1553
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
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1690
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2721565, upper bound: 29.3083726
time: 67.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2740850, upper bound: 29.3420240
time: 71.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -45.9694901, 30.0432625, -46.2052422, 30.1182861, -76.0877762, 76.2485046
1: -27.5187378, 28.5110703, -27.7061138, 28.5997009, -56.1184387, 56.2171822
2: -24.3301220, 25.2701073, -24.5286617, 25.3627758, -49.6928940, 49.7987671
3: -30.4228592, 28.2682495, -30.6447296, 28.3859043, -58.8087616, 58.9129791
4: -28.8628693, 26.9116955, -29.0872307, 27.0157204, -55.8785896, 55.9989243
5: -30.0093632, 30.7595882, -30.2101612, 30.8519783, -60.8613434, 60.9697495
6: -36.1457329, 22.6007729, -36.2505836, 22.7876396, -58.9333725, 58.8513565
7: -35.3772812, 31.7490311, -35.5692482, 31.8493462, -67.2266235, 67.3182831
8: -38.1744003, 29.5579147, -38.4453964, 29.7162418, -67.8906403, 68.0033112
9: -31.6431274, 26.8167267, -31.7437744, 26.8768044, -58.5199318, 58.5604973
10: -43.6432533, 37.5453224, -43.7636337, 37.6828041, -81.3260574, 81.3089600
11: -42.1729164, 32.9116898, -42.2812233, 33.0666428, -75.2395554, 75.1929092
12: -37.4220657, 35.3014412, -37.5326843, 35.5100822, -72.9321442, 72.8341217
13: -39.9095535, 43.2364388, -40.1105576, 43.3429832, -83.2525330, 83.3470001
14: -65.6902771, 23.1171322, -65.8798294, 23.2341862, -88.9244614, 88.9969635
15: -36.4652557, 24.9646683, -36.6322632, 25.0519810, -61.5172348, 61.5969276
16: -45.5830193, 31.4142246, -45.7091751, 31.5820866, -77.1651077, 77.1233978
17: -65.4221497, 41.4899597, -65.5675201, 41.6618042, -107.0839539, 107.0574646
18: -37.5605469, 33.2250214, -37.7156601, 33.5802193, -71.1407623, 70.9406815
19: -31.6210709, 20.3035583, -31.7070770, 20.4157143, -52.0367813, 52.0106354
20: -27.1066837, 20.6362514, -27.1726608, 20.7533875, -47.8600693, 47.8089142
21: -38.0594177, 23.4904633, -38.1596069, 23.5929184, -61.6523361, 61.6500626
22: -37.2206306, 24.8171177, -37.3522491, 24.8993759, -62.1200066, 62.1693649
23: -31.3077526, 22.3109169, -31.3861504, 22.4328098, -53.7405548, 53.6970673
24: -32.3063812, 23.5175591, -32.4342918, 23.6453762, -55.9517517, 55.9518509
25: -29.7022305, 28.0972710, -29.7969532, 28.2141685, -57.9163971, 57.8942261
26: -46.3497162, 33.2999458, -46.4563637, 33.5284500, -79.8781586, 79.7563019
27: -38.3113251, 22.2218895, -38.4135933, 22.3379078, -60.6492157, 60.6354828
28: -31.5238132, 25.7943783, -31.5932770, 25.9337978, -57.4576111, 57.3876572
29: -38.6841278, 24.5741920, -38.8352432, 24.6854591, -63.3695831, 63.4094353
30: -37.4048538, 26.1169167, -37.5077057, 26.3222580, -63.7271118, 63.6246223
31: -35.8362617, 25.3846893, -35.9823380, 25.5381165, -61.3743782, 61.3670197
32: -33.0529976, 25.0866852, -33.1561050, 25.2051716, -58.2581711, 58.2427864
33: -54.5128250, 33.1290245, -54.6469040, 33.2687073, -87.7815323, 87.7759247
34: -47.1580200, 23.4642315, -47.2613602, 23.6259708, -70.7839813, 70.7255859
35: -43.0807037, 27.9668007, -43.1891861, 28.0767574, -71.1574554, 71.1559830
36: -41.3091202, 31.1018734, -41.3934097, 31.1805897, -72.4897079, 72.4952850
37: -60.6097794, 27.9976120, -60.7974319, 28.2688713, -88.8786469, 88.7950439
38: -52.8992310, 40.5477066, -53.0411568, 40.6816940, -93.5809250, 93.5888672
39: -61.9169426, 34.6874008, -62.0894241, 34.7544594, -96.6714020, 96.7768173
40: -54.5733223, 23.2231045, -54.7417297, 23.5009689, -78.0742950, 77.9648285
41: -35.3241119, 21.5688629, -35.4205704, 21.6966209, -57.0207329, 56.9894333
42: -29.6255627, 19.8490582, -29.6851463, 19.9845810, -49.6101379, 49.5342026

Time for backsubstitution: 1.87 seconds

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
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1629
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
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1762
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
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1713
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
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1623
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
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1690
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

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3037336, upper bound: 29.2404275
time: 84.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3055411, upper bound: 29.2743247
time: 82.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -46.0382881, 30.0561180, -46.3716354, 30.2254505, -76.2637329, 76.4277496
1: -27.5620594, 28.5211067, -27.8121586, 28.6505203, -56.2125778, 56.3332672
2: -24.3904228, 25.2796974, -24.6670151, 25.4528961, -49.8433151, 49.9467125
3: -30.4980278, 28.2821693, -30.8159332, 28.5125885, -59.0106163, 59.0981026
4: -28.9356632, 26.9233437, -29.2512379, 27.1266785, -56.0623398, 56.1745834
5: -30.0746613, 30.7727089, -30.3612633, 30.9871025, -61.0617638, 61.1339645
6: -36.1640472, 22.6426964, -36.3209839, 22.8956547, -59.0597000, 58.9636803
7: -35.4242172, 31.7613392, -35.6951294, 31.9076633, -67.3318787, 67.4564667
8: -38.2509460, 29.5710392, -38.6198349, 29.8205452, -68.0714798, 68.1908722
9: -31.6854439, 26.8408642, -31.8463879, 26.9658585, -58.6513023, 58.6872444
10: -43.6696472, 37.6128540, -43.8550224, 37.8638153, -81.5334625, 81.4678802
11: -42.1936264, 33.0204315, -42.4584351, 33.3088493, -75.5024719, 75.4788666
12: -37.4394455, 35.4342117, -37.7606087, 35.8060837, -73.2455139, 73.1948166
13: -39.9253616, 43.2644730, -40.1665344, 43.4291763, -83.3545303, 83.4310074
14: -65.7272644, 23.1909599, -66.0431442, 23.4004784, -89.1277466, 89.2341003
15: -36.5391502, 24.9797192, -36.8130188, 25.1231594, -61.6623077, 61.7927322
16: -45.6114578, 31.4629021, -45.8110695, 31.7200508, -77.3314972, 77.2739716
17: -65.4512482, 41.6412277, -65.8175964, 41.9967155, -107.4479370, 107.4588242
18: -37.5818291, 33.2749634, -37.7999496, 33.7012367, -71.2830658, 71.0749130
19: -31.6389599, 20.3565445, -31.8282280, 20.5358086, -52.1747589, 52.1847687
20: -27.1252766, 20.6796246, -27.2737141, 20.8536434, -47.9789200, 47.9533386
21: -38.0788116, 23.5630760, -38.3100548, 23.7580223, -61.8368340, 61.8731308
22: -37.2426147, 24.8848057, -37.4516068, 25.0567513, -62.2993622, 62.3364105
23: -31.3232498, 22.3476925, -31.4768333, 22.5186787, -53.8419266, 53.8245239
24: -32.3250122, 23.5306511, -32.4994888, 23.6833458, -56.0083542, 56.0301361
25: -29.7201042, 28.1249294, -29.8671932, 28.2899590, -58.0100632, 57.9921227
26: -46.3716011, 33.3956718, -46.6480217, 33.7423668, -80.1139679, 80.0436935
27: -38.3323364, 22.2381382, -38.4871864, 22.3832169, -60.7155457, 60.7253189
28: -31.5399418, 25.8299999, -31.6872711, 26.0168495, -57.5567856, 57.5172729
29: -38.7022781, 24.6648998, -38.9665642, 24.8842926, -63.5865631, 63.6314621
30: -37.4200859, 26.1695843, -37.5852356, 26.4474277, -63.8675156, 63.7548103
31: -35.8572426, 25.4252434, -36.0845413, 25.6349220, -61.4921646, 61.5097771
32: -33.0695724, 25.1373806, -33.2430763, 25.3242989, -58.3938599, 58.3804550
33: -54.5816383, 33.1481743, -54.8189926, 33.3688583, -87.9505005, 87.9671631
34: -47.2217941, 23.4806232, -47.4129105, 23.7316036, -70.9533844, 70.8935242
35: -43.1281624, 27.9777336, -43.3084526, 28.1291466, -71.2573090, 71.2861862
36: -41.3265762, 31.1425858, -41.4616547, 31.2816620, -72.6082306, 72.6042404
37: -60.6370506, 28.0395203, -60.8972626, 28.3778400, -89.0148773, 88.9367828
38: -52.9373970, 40.5741806, -53.1631088, 40.7641373, -93.7015381, 93.7372894
39: -61.9631615, 34.6976700, -62.2156944, 34.8124962, -96.7756500, 96.9133606
40: -54.6212044, 23.2365971, -54.8692360, 23.5585613, -78.1797638, 78.1058350
41: -35.3451004, 21.6008453, -35.4902039, 21.7811642, -57.1262665, 57.0910454
42: -29.6460361, 19.9057713, -29.7609406, 20.1176834, -49.7637177, 49.6667099

Time for backsubstitution: 1.89 seconds

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
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1623
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
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1690
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

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3042599, upper bound: 29.2781357
time: 163.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3060802, upper bound: 29.3120592
time: 90.74 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 255.76 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 255.76
Output dim: 2, lower bound: -29.2717088, upper bound: 29.2392394
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 255.76
Output dim: 2, lower bound: -29.2717088, upper bound: 29.2731594
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 255.76
Output dim: 2, lower bound: -29.2721565, upper bound: 29.2768778
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 255.76
Output dim: 2, lower bound: -29.2740850, upper bound: 29.3108072
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 255.76
Output dim: 2, lower bound: -29.2717088, upper bound: 29.2707280
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 255.76
Output dim: 2, lower bound: -29.2735856, upper bound: 29.3044100
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 255.76
Output dim: 2, lower bound: -29.2721565, upper bound: 29.3083726
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 255.76
Output dim: 2, lower bound: -29.2740850, upper bound: 29.3420240
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 255.76
Output dim: 2, lower bound: -29.3037336, upper bound: 29.2404275
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 255.76
Output dim: 2, lower bound: -29.3055411, upper bound: 29.2743247
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 255.76
Output dim: 2, lower bound: -29.3042599, upper bound: 29.2781357
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 255.76
Output dim: 2, lower bound: -29.3060802, upper bound: 29.3120592
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 255.76
Output dim: 2, lower bound: -29.3069861, upper bound: 29.3069979
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 255.76
Output dim: 2, lower bound: -29.3074997, upper bound: 29.3446372
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 255.76
Output dim: 2, lower bound: -29.2750173, upper bound: 29.2755156
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 255.76
Output dim: 2, lower bound: -29.3136875, upper bound: 29.3132035
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 255.76
Output dim: 2, lower bound: -29.2750173, upper bound: 29.3067859
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 255.76
Output dim: 2, lower bound: -29.3136875, upper bound: 29.3444516
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 255.76
Output dim: 2, lower bound: -29.3449520, upper bound: 29.2767014
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 255.76
Output dim: 2, lower bound: -29.3456274, upper bound: 29.3144508
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 255.76
Output dim: 2, lower bound: -29.3449520, upper bound: 29.3079366
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 255.76
Output dim: 2, lower bound: -29.3456274, upper bound: 29.3456277

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 77.37 + 3596.68 = 3674.05 seconds
