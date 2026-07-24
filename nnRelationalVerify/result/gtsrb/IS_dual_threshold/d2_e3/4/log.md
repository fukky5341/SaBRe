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
execution time: IAR + RelationalAnalysis = 2.86 + 73.03 = 75.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -29.3529070, upper bound: 29.3529070

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1657

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3498756, upper bound: 29.3136239
time: 70.88 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3498756, upper bound: 29.3503747
time: 91.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 162.74 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 162.74
Output dim: 2, lower bound: -29.3498756, upper bound: 29.3136239
IS_B2, status: Status.UNKNOWN, split count: 1, time: 162.74
Output dim: 2, lower bound: -29.3498756, upper bound: 29.3503747

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -46.3269386, 30.1964340, -46.2311020, 30.1737118, -76.5006485, 76.4275360
1: -27.7416058, 28.6856213, -27.6831894, 28.6672592, -56.4088669, 56.3688126
2: -24.5956917, 25.4527054, -24.5106277, 25.4348145, -50.0305023, 49.9633293
3: -30.7288380, 28.5096378, -30.6282845, 28.4859524, -59.2147827, 59.1379166
4: -29.1614685, 27.1125221, -29.0639324, 27.0908699, -56.2523346, 56.1764488
5: -30.2811756, 30.9463463, -30.1942825, 30.9242840, -61.2054596, 61.1406212
6: -36.3631363, 22.8314190, -36.3314590, 22.7706795, -59.1338120, 59.1628799
7: -35.6173248, 31.9550476, -35.5500145, 31.9322891, -67.5496140, 67.5050659
8: -38.4894562, 29.8766708, -38.3992462, 29.8532219, -68.3426819, 68.2759171
9: -31.8324184, 26.9588528, -31.7515831, 26.9218063, -58.7542267, 58.7104340
10: -43.8502197, 37.8116837, -43.7914429, 37.7190399, -81.5692596, 81.6031265
11: -42.3816490, 33.2695389, -42.3473701, 33.0952606, -75.4769135, 75.6169128
12: -37.6513672, 35.7337646, -37.6185265, 35.5738602, -73.2252197, 73.3522949
13: -40.1082344, 43.4712677, -40.0420151, 43.4310570, -83.5392914, 83.5132828
14: -65.9302597, 23.4248314, -65.8653870, 23.2943935, -89.2246552, 89.2902222
15: -36.7925835, 25.1426811, -36.6851578, 25.1171913, -61.9097748, 61.8278351
16: -45.8488922, 31.6687050, -45.7908897, 31.5736485, -77.4225311, 77.4595947
17: -65.6553116, 42.0185204, -65.6022949, 41.7706566, -107.4259644, 107.6208115
18: -37.8647575, 33.5592041, -37.8251152, 33.4482079, -71.3129654, 71.3843231
19: -31.7885876, 20.4983177, -31.7587700, 20.4123974, -52.2009850, 52.2570877
20: -27.2362595, 20.8364239, -27.2045765, 20.7657623, -48.0020218, 48.0410004
21: -38.2334976, 23.7386208, -38.1974220, 23.6177330, -61.8512268, 61.9360390
22: -37.4170609, 25.0542660, -37.3763275, 24.9505577, -62.3676147, 62.4305954
23: -31.4563198, 22.4771385, -31.4304905, 22.4067230, -53.8630371, 53.9076271
24: -32.5232010, 23.6289482, -32.4900284, 23.5849915, -56.1081848, 56.1189766
25: -29.8697186, 28.2603130, -29.8357105, 28.2048340, -58.0745544, 58.0960236
26: -46.5624237, 33.6600800, -46.5220337, 33.5067139, -80.0691376, 80.1821136
27: -38.4728622, 22.3765774, -38.4335136, 22.3045444, -60.7774048, 60.8100891
28: -31.6607361, 25.9735622, -31.6339169, 25.9012852, -57.5620193, 57.6074753
29: -38.8928642, 24.8916645, -38.8564262, 24.7493362, -63.6421890, 63.7480927
30: -37.5632858, 26.3812065, -37.5331688, 26.2718887, -63.8351746, 63.9143715
31: -36.1084061, 25.5925770, -36.0723648, 25.5193901, -61.6277962, 61.6649323
32: -33.2034760, 25.2864017, -33.1667557, 25.2297440, -58.4332199, 58.4531555
33: -54.8534355, 33.2777100, -54.7312202, 33.2450638, -88.0984955, 88.0089264
34: -47.4569855, 23.6038933, -47.3706665, 23.5776863, -71.0346680, 70.9745636
35: -43.3455429, 28.0708733, -43.2634239, 28.0517731, -71.3973160, 71.3342972
36: -41.4595337, 31.2385159, -41.4158592, 31.1933556, -72.6528778, 72.6543732
37: -60.9709816, 28.2357941, -60.9180145, 28.1857872, -89.1567688, 89.1538086
38: -53.1818657, 40.7120399, -53.0967026, 40.6801758, -93.8620453, 93.8087311
39: -62.2438049, 34.7926331, -62.1363487, 34.7681122, -97.0119171, 96.9289856
40: -54.9476395, 23.4282570, -54.8767090, 23.4062901, -78.3539124, 78.3049622
41: -35.5272102, 21.7252693, -35.4865417, 21.6732903, -57.2005005, 57.2118111
42: -29.7681217, 20.0689621, -29.7344437, 19.9933186, -49.7614403, 49.8034058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.3102405, upper bound: 29.3108082
time: 83.48 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.3102405, upper bound: 29.3108082
time: 74.41 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -46.3955994, 30.2111759, -46.4302940, 30.3163681, -76.7119675, 76.6414719
1: -27.7837753, 28.6968994, -27.8045044, 28.7355576, -56.5193329, 56.5014038
2: -24.6570854, 25.4636993, -24.6677361, 25.5680847, -50.2251701, 50.1314278
3: -30.8012848, 28.5238876, -30.8187580, 28.6495171, -59.4508018, 59.3426437
4: -29.2361279, 27.1265202, -29.2466602, 27.2379017, -56.4740295, 56.3731766
5: -30.3436375, 30.9598808, -30.3600311, 31.1030064, -61.4466438, 61.3199081
6: -36.3839378, 22.8584042, -36.4300728, 22.8665295, -59.2504654, 59.2884750
7: -35.6661758, 31.9672680, -35.6982651, 32.0031662, -67.6693420, 67.6655273
8: -38.5550842, 29.8903294, -38.5709686, 29.9573803, -68.5124664, 68.4612961
9: -31.8909721, 26.9820175, -31.9051304, 27.0509930, -58.9419556, 58.8871460
10: -43.8902664, 37.8749046, -43.9299355, 37.9399261, -81.8301926, 81.8048401
11: -42.4044456, 33.3943787, -42.6340981, 33.4040108, -75.8084564, 76.0284653
12: -37.6704178, 35.8490753, -37.8868828, 35.8626938, -73.5331039, 73.7359543
13: -40.1349449, 43.4974403, -40.1375580, 43.5608406, -83.6957855, 83.6349945
14: -65.9719009, 23.5190735, -66.1376877, 23.5262127, -89.4981079, 89.6567535
15: -36.8667107, 25.1597233, -36.8896637, 25.2123814, -62.0790939, 62.0493851
16: -45.8878250, 31.7169380, -45.9419441, 31.7512550, -77.6390839, 77.6588745
17: -65.6883087, 42.1998787, -66.0121231, 42.2142410, -107.9025497, 108.2119980
18: -37.8871765, 33.6388626, -38.0033607, 33.6522903, -71.5394669, 71.6422272
19: -31.8084641, 20.5607567, -31.9460506, 20.5704384, -52.3788910, 52.5068054
20: -27.2569122, 20.8868542, -27.3623199, 20.8951073, -48.1520195, 48.2491760
21: -38.2574615, 23.8260155, -38.4488335, 23.8324127, -62.0898743, 62.2748451
22: -37.4405098, 25.1304932, -37.5747833, 25.1489677, -62.5894775, 62.7052727
23: -31.4733372, 22.5264969, -31.5819378, 22.5312996, -54.0046387, 54.1084366
24: -32.5426903, 23.6598892, -32.6270447, 23.6712685, -56.2139587, 56.2869339
25: -29.8903904, 28.3010883, -29.9524403, 28.3156643, -58.2060547, 58.2535286
26: -46.5843163, 33.7705994, -46.8389015, 33.7768974, -80.3612137, 80.6094971
27: -38.4976196, 22.4292011, -38.6256256, 22.4340496, -60.9316711, 61.0548248
28: -31.6786785, 26.0253162, -31.8013630, 26.0314941, -57.7101746, 57.8266792
29: -38.9139557, 24.9965420, -39.1041641, 25.0006599, -63.9146156, 64.1007080
30: -37.5825691, 26.4581223, -37.7127304, 26.4684219, -64.0509949, 64.1708527
31: -36.1320190, 25.6459732, -36.2603683, 25.6570282, -61.7890472, 61.9063416
32: -33.2259064, 25.3189602, -33.2728424, 25.3321171, -58.5580139, 58.5918007
33: -54.9373436, 33.2986794, -54.9629555, 33.4308548, -88.3681946, 88.2616272
34: -47.5178719, 23.6212578, -47.5406189, 23.6959076, -71.2137756, 71.1618729
35: -43.4006424, 28.0826054, -43.4226265, 28.1297073, -71.5303421, 71.5052338
36: -41.4820061, 31.2715855, -41.5183716, 31.2932472, -72.7752380, 72.7899551
37: -61.0038147, 28.2647705, -61.0610771, 28.2854042, -89.2892151, 89.3258514
38: -53.2372055, 40.7313156, -53.2884483, 40.7716331, -94.0088348, 94.0197601
39: -62.3157425, 34.8071175, -62.3398361, 34.9335403, -97.2492828, 97.1469498
40: -54.9944420, 23.4412556, -55.0316200, 23.4878178, -78.4822617, 78.4728775
41: -35.5534401, 21.7470703, -35.5860100, 21.7613983, -57.3148346, 57.3330727
42: -29.7917328, 20.1241016, -29.8313961, 20.1396427, -49.9313736, 49.9554977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=479, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1673

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3103073, upper bound: 29.3473413
time: 83.06 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3479594, upper bound: 29.3479593
time: 164.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 249.74 seconds
IS_B1_A1, status: Status.VERIFIED, split count: 2, time: 249.74
Output dim: 2, lower bound: -29.3102405, upper bound: 29.3108082
IS_B1_A2, status: Status.VERIFIED, split count: 2, time: 249.74
Output dim: 2, lower bound: -29.3102405, upper bound: 29.3108082
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 249.74
Output dim: 2, lower bound: -29.3103073, upper bound: 29.3473413
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 249.74
Output dim: 2, lower bound: -29.3479594, upper bound: 29.3479593

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -46.2340126, 30.1799355, -46.3508568, 30.3011036, -76.5351105, 76.5307922
1: -27.6842575, 28.6719780, -27.7546558, 28.7231846, -56.4074402, 56.4266319
2: -24.5163002, 25.4398270, -24.5983505, 25.5563622, -50.0726624, 50.0381737
3: -30.6263103, 28.4889565, -30.7326660, 28.6322994, -59.2586098, 59.2216225
4: -29.0662384, 27.0967522, -29.1628494, 27.2232914, -56.2895279, 56.2596016
5: -30.1908188, 30.9265385, -30.2849617, 31.0866051, -61.2774200, 61.2115021
6: -36.3375473, 22.7399120, -36.4071732, 22.8071060, -59.1446533, 59.1470833
7: -35.5474167, 31.9362411, -35.6398544, 31.9877300, -67.5351410, 67.5760956
8: -38.3781662, 29.8561764, -38.4838066, 29.9405270, -68.3186951, 68.3399811
9: -31.7920208, 26.9174938, -31.8564434, 27.0194988, -58.8115196, 58.7739372
10: -43.8192673, 37.7137833, -43.8946533, 37.8603210, -81.6795883, 81.6084366
11: -42.3531494, 33.1426697, -42.6089897, 33.2799454, -75.6330948, 75.7516556
12: -37.6256561, 35.5420532, -37.8648911, 35.7118034, -73.3374634, 73.4069443
13: -40.0868492, 43.4151688, -40.1135330, 43.5203018, -83.6071472, 83.5287018
14: -65.8818512, 23.3502731, -66.0934067, 23.4425125, -89.3243637, 89.4436798
15: -36.6815643, 25.1227989, -36.7977791, 25.1941452, -61.8756866, 61.9205704
16: -45.7990913, 31.5753956, -45.8972168, 31.6809464, -77.4800415, 77.4726105
17: -65.6152267, 41.8522186, -65.9764099, 42.0434418, -107.6586609, 107.8286285
18: -37.8309860, 33.5197945, -37.9761124, 33.5935287, -71.4245148, 71.4959030
19: -31.7648468, 20.4389439, -31.9246178, 20.5105057, -52.2753525, 52.3635597
20: -27.2111111, 20.7858047, -27.3400269, 20.8454514, -48.0565567, 48.1258316
21: -38.2095642, 23.6589165, -38.4253845, 23.7503052, -61.9598694, 62.0842972
22: -37.3829803, 24.9806194, -37.5465775, 25.0739040, -62.4568787, 62.5271988
23: -31.4355145, 22.4403400, -31.5633163, 22.4887714, -53.9242859, 54.0036545
24: -32.4918976, 23.6269493, -32.6020660, 23.6548519, -56.1467514, 56.2290154
25: -29.8446541, 28.2290077, -29.9298210, 28.2798138, -58.1244659, 58.1588287
26: -46.5277824, 33.5487595, -46.8108864, 33.6676903, -80.1954727, 80.3596497
27: -38.4449615, 22.3807678, -38.6000633, 22.4100361, -60.8549957, 60.9808311
28: -31.6396294, 25.9423847, -31.7821445, 25.9906292, -57.6302567, 57.7245255
29: -38.8642807, 24.7901573, -39.0796509, 24.8986130, -63.7628937, 63.8698082
30: -37.5433769, 26.3339176, -37.6934891, 26.4070492, -63.9504242, 64.0274048
31: -36.0808487, 25.5516415, -36.2352676, 25.6105022, -61.6913528, 61.7869110
32: -33.1830292, 25.1982975, -33.2515793, 25.2727623, -58.4557800, 58.4498749
33: -54.7736893, 33.2495956, -54.8823433, 33.4067383, -88.1804199, 88.1319427
34: -47.3700905, 23.5802708, -47.4680252, 23.6758595, -71.0459442, 71.0482941
35: -43.2859573, 28.0536118, -43.3659897, 28.1153927, -71.4013519, 71.4196014
36: -41.4378471, 31.1762028, -41.4962921, 31.2459183, -72.6837616, 72.6724930
37: -60.9320297, 28.1653252, -61.0254326, 28.2362022, -89.1682281, 89.1907578
38: -53.1355171, 40.6643753, -53.2378349, 40.7386246, -93.8741455, 93.9022064
39: -62.2026825, 34.7680740, -62.2838631, 34.9146881, -97.1173706, 97.0519333
40: -54.8759995, 23.4055462, -54.9729004, 23.4699497, -78.3459473, 78.3784485
41: -35.5019684, 21.6576920, -35.5605469, 21.7167759, -57.2187347, 57.2182312
42: -29.7429333, 19.9881458, -29.8072453, 20.0723343, -49.8152695, 49.7953911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=479, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2690194, upper bound: 29.3064375
time: 86.51 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2690194, upper bound: 29.3443766
time: 79.51 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -46.4017906, 30.2874851, -46.4211349, 30.3142948, -76.7160873, 76.7086182
1: -27.7911568, 28.7233067, -27.7989216, 28.7337246, -56.5248795, 56.5222244
2: -24.6560268, 25.5301208, -24.6601181, 25.5661526, -50.2221756, 50.1902351
3: -30.7988682, 28.6160984, -30.8093300, 28.6466236, -59.4454918, 59.4254303
4: -29.2318478, 27.2081356, -29.2375259, 27.2353287, -56.4671707, 56.4456596
5: -30.3430805, 31.0620823, -30.3515167, 31.1001263, -61.4431992, 61.4135971
6: -36.4083672, 22.8551636, -36.4259872, 22.8554249, -59.2637939, 59.2811508
7: -35.6759491, 31.9950905, -35.6893806, 32.0007248, -67.6766739, 67.6844635
8: -38.5539589, 29.9608841, -38.5619049, 29.9540482, -68.5080109, 68.5227890
9: -31.8954430, 27.0075531, -31.8997326, 27.0440369, -58.9394798, 58.9072876
10: -43.9125061, 37.8966637, -43.9235344, 37.9300690, -81.8425751, 81.8201828
11: -42.5308723, 33.3870850, -42.6301155, 33.3912125, -75.9220734, 76.0171967
12: -37.8540802, 35.8403091, -37.8827515, 35.8469810, -73.7010651, 73.7230606
13: -40.1439438, 43.5020103, -40.1306381, 43.5488892, -83.6928253, 83.6326447
14: -66.0464401, 23.5177536, -66.1314316, 23.5176220, -89.5640564, 89.6491852
15: -36.8659363, 25.1942940, -36.8762283, 25.2095585, -62.0754929, 62.0705223
16: -45.9035149, 31.7153206, -45.9284096, 31.7318096, -77.6353226, 77.6437225
17: -65.8659515, 42.1896095, -66.0060425, 42.1974068, -108.0633545, 108.1956482
18: -37.9158707, 33.6419754, -37.9974251, 33.6448975, -71.5607605, 71.6394043
19: -31.8865299, 20.5598660, -31.9429836, 20.5644798, -52.4510117, 52.5028496
20: -27.3127213, 20.8868294, -27.3589668, 20.8896923, -48.2024078, 48.2457924
21: -38.3604965, 23.8252544, -38.4451752, 23.8243103, -62.1848068, 62.2704277
22: -37.4831467, 25.1388988, -37.5694199, 25.1424999, -62.6256485, 62.7083206
23: -31.5267334, 22.5270214, -31.5793324, 22.5265198, -54.0532455, 54.1063538
24: -32.5581055, 23.6654091, -32.6212921, 23.6685658, -56.2266693, 56.2867012
25: -29.9156055, 28.3071938, -29.9483852, 28.3103485, -58.2259521, 58.2555771
26: -46.7204170, 33.7640381, -46.8339005, 33.7648544, -80.4852600, 80.5979385
27: -38.5192108, 22.4287643, -38.6212654, 22.4285126, -60.9477234, 61.0500298
28: -31.7339954, 26.0261898, -31.7986565, 26.0270481, -57.7610435, 57.8248444
29: -38.9964981, 24.9902763, -39.0987053, 24.9906464, -63.9871445, 64.0889816
30: -37.6213074, 26.4605865, -37.7090454, 26.4613743, -64.0826797, 64.1696320
31: -36.1836586, 25.6493645, -36.2566681, 25.6522141, -61.8358612, 61.9060326
32: -33.2704620, 25.3186760, -33.2688026, 25.3246536, -58.5951080, 58.5874786
33: -54.9474182, 33.3503036, -54.9528847, 33.4265518, -88.3739700, 88.3031921
34: -47.5226402, 23.6863327, -47.5328674, 23.6926289, -71.2152710, 71.2191925
35: -43.4066277, 28.1062012, -43.4149628, 28.1265831, -71.5332031, 71.5211563
36: -41.5068970, 31.2779408, -41.5148354, 31.2872391, -72.7941360, 72.7927704
37: -61.0332413, 28.2752457, -61.0545044, 28.2791786, -89.3124237, 89.3297501
38: -53.2604904, 40.7474747, -53.2786598, 40.7656288, -94.0261230, 94.0261307
39: -62.3304863, 34.8265877, -62.3317871, 34.9248428, -97.2553253, 97.1583710
40: -55.0049210, 23.4638805, -55.0224266, 23.4840889, -78.4890060, 78.4862976
41: -35.5722847, 21.7448044, -35.5823135, 21.7514458, -57.3237305, 57.3271103
42: -29.8191261, 20.1231766, -29.8282642, 20.1313839, -49.9505081, 49.9514389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=478, inp2_unstable=479, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3089804, upper bound: 29.3453746
time: 69.45 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3453746, upper bound: 29.3453746
time: 73.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 145.60 seconds
IS_B2_A1_A1, status: Status.VERIFIED, split count: 3, time: 145.60
Output dim: 2, lower bound: -29.2690194, upper bound: 29.3064375
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 145.60
Output dim: 2, lower bound: -29.2690194, upper bound: 29.3443766
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 145.60
Output dim: 2, lower bound: -29.3089804, upper bound: 29.3453746
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 145.60
Output dim: 2, lower bound: -29.3453746, upper bound: 29.3453746

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -46.2416229, 30.2663803, -46.3410721, 30.2990017, -76.5406265, 76.6074448
1: -27.6847248, 28.7119179, -27.7463074, 28.7212296, -56.4059525, 56.4582253
2: -24.5104542, 25.5397263, -24.5880432, 25.5545120, -50.0649643, 50.1277695
3: -30.6185799, 28.6054935, -30.7210236, 28.6293240, -59.2478943, 59.3265076
4: -29.0622501, 27.1761818, -29.1523972, 27.2208004, -56.2830505, 56.3285789
5: -30.1858273, 31.0629730, -30.2737732, 31.0839844, -61.2698135, 61.3367424
6: -36.3726349, 22.7331028, -36.4034424, 22.7929039, -59.1655388, 59.1365433
7: -35.5532227, 31.9989624, -35.6272354, 31.9845963, -67.5378189, 67.6261978
8: -38.3759232, 29.9477062, -38.4729385, 29.9376812, -68.3135986, 68.4206390
9: -31.8021259, 26.9333420, -31.8529530, 27.0108604, -58.8129807, 58.7862930
10: -43.9028015, 37.7212753, -43.8898849, 37.8447418, -81.7475433, 81.6111603
11: -42.4628563, 33.1342621, -42.6048470, 33.2663879, -75.7292480, 75.7391052
12: -37.8630600, 35.5303879, -37.8609924, 35.6921310, -73.5551910, 73.3913727
13: -40.0842819, 43.4442635, -40.0999832, 43.5160599, -83.6003418, 83.5442429
14: -65.9982910, 23.3477116, -66.0859909, 23.4311142, -89.4293976, 89.4337006
15: -36.6732445, 25.1612530, -36.7791290, 25.1906586, -61.8638840, 61.9403763
16: -45.8245888, 31.5741158, -45.8866463, 31.6648140, -77.4894028, 77.4607620
17: -65.7698441, 41.8435364, -65.9711609, 42.0268822, -107.7967224, 107.8146973
18: -37.9074478, 33.5186844, -37.9720573, 33.5839005, -71.4913483, 71.4907379
19: -31.8298626, 20.4380550, -31.9216881, 20.5049324, -52.3347931, 52.3597412
20: -27.2751617, 20.7871342, -27.3365841, 20.8400421, -48.1152039, 48.1237183
21: -38.3134460, 23.6603088, -38.4219894, 23.7412872, -62.0547333, 62.0822983
22: -37.4338837, 24.9845200, -37.5417786, 25.0625687, -62.4964447, 62.5262985
23: -31.4772491, 22.4450493, -31.5605106, 22.4839592, -53.9612083, 54.0055542
24: -32.5073013, 23.6317406, -32.5962906, 23.6516190, -56.1589203, 56.2280312
25: -29.8676777, 28.2368984, -29.9253025, 28.2734489, -58.1411209, 58.1621971
26: -46.7150726, 33.5410004, -46.8057442, 33.6507416, -80.3658142, 80.3467407
27: -38.4698830, 22.3760605, -38.5957718, 22.4028893, -60.8727722, 60.9718285
28: -31.6789703, 25.9464016, -31.7794266, 25.9870071, -57.6659698, 57.7258301
29: -38.9252281, 24.7860794, -39.0751762, 24.8878384, -63.8130569, 63.8612556
30: -37.5757713, 26.3367691, -37.6901016, 26.3984337, -63.9742050, 64.0268707
31: -36.1387711, 25.5549622, -36.2314110, 25.6051216, -61.7438927, 61.7863731
32: -33.2475014, 25.1971569, -33.2480621, 25.2655525, -58.5130539, 58.4452209
33: -54.7840881, 33.3145332, -54.8727722, 33.4030380, -88.1871262, 88.1873016
34: -47.3823776, 23.6182442, -47.4615555, 23.6724510, -71.0548248, 71.0798035
35: -43.2900391, 28.0931244, -43.3584900, 28.1132298, -71.4032669, 71.4516144
36: -41.4585571, 31.1794243, -41.4909897, 31.2395706, -72.6981277, 72.6704102
37: -60.9683609, 28.1743660, -61.0193977, 28.2291718, -89.1975250, 89.1937637
38: -53.1618996, 40.6787872, -53.2293205, 40.7337799, -93.8956757, 93.9081116
39: -62.2218857, 34.7968292, -62.2753944, 34.9119682, -97.1338501, 97.0722198
40: -54.8979607, 23.4167862, -54.9647942, 23.4651680, -78.3631287, 78.3815765
41: -35.5277252, 21.6573849, -35.5568161, 21.7069302, -57.2346573, 57.2141991
42: -29.7813587, 19.9905033, -29.8044090, 20.0625401, -49.8438988, 49.7949104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=478, inp2_unstable=479, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2677246, upper bound: 29.2901069
time: 77.77 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2677247, upper bound: 29.3051341
time: 80.18 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -46.3011093, 30.2675972, -46.3799515, 30.3062191, -76.6073227, 76.6475525
1: -27.7230434, 28.7075729, -27.7710342, 28.7273216, -56.4503632, 56.4786034
2: -24.5365887, 25.5157127, -24.6115322, 25.5602856, -50.0968628, 50.1272430
3: -30.6683731, 28.5935898, -30.7561855, 28.6374817, -59.3058548, 59.3497772
4: -29.1109905, 27.1891804, -29.1882648, 27.2275810, -56.3385696, 56.3774452
5: -30.2147980, 31.0412617, -30.2993088, 31.0916252, -61.3064232, 61.3405685
6: -36.3761368, 22.8033791, -36.4128036, 22.8343067, -59.2104416, 59.2161789
7: -35.5670547, 31.9753418, -35.6445770, 31.9925919, -67.5596390, 67.6199188
8: -38.4270248, 29.9402905, -38.5102081, 29.9456997, -68.3727264, 68.4505005
9: -31.8658943, 26.9308968, -31.8876591, 27.0129375, -58.8788223, 58.8185539
10: -43.8697395, 37.7201309, -43.9061317, 37.8581505, -81.7278900, 81.6262589
11: -42.4965019, 33.2300415, -42.6161232, 33.3273087, -75.8237915, 75.8461609
12: -37.8253746, 35.5953598, -37.8710480, 35.7473297, -73.5727005, 73.4664001
13: -40.1046753, 43.4570122, -40.1146469, 43.5306091, -83.6352844, 83.5716553
14: -65.9808273, 23.3723431, -66.1047516, 23.4583740, -89.4392014, 89.4770889
15: -36.7715683, 25.1645069, -36.8382492, 25.1974182, -61.9689865, 62.0027542
16: -45.8525505, 31.6179733, -45.9078178, 31.6919689, -77.5445175, 77.5257874
17: -65.8215179, 41.9860840, -65.9879761, 42.1145248, -107.9360352, 107.9740601
18: -37.8774796, 33.5378876, -37.9822693, 33.6025314, -71.4799957, 71.5201569
19: -31.8575382, 20.5018349, -31.9311600, 20.5408783, -52.3984146, 52.4329948
20: -27.2800465, 20.8294239, -27.3457603, 20.8663101, -48.1463547, 48.1751862
21: -38.3295479, 23.7208138, -38.4326019, 23.7818470, -62.1113968, 62.1534157
22: -37.4489594, 25.0346527, -37.5555038, 25.1000748, -62.5490265, 62.5901489
23: -31.5007439, 22.4775696, -31.5687160, 22.5064030, -54.0071487, 54.0462875
24: -32.5196533, 23.6474457, -32.6057205, 23.6612625, -56.1809082, 56.2531662
25: -29.8886433, 28.2628479, -29.9374008, 28.2922897, -58.1809235, 58.2002487
26: -46.6838913, 33.5796700, -46.8190460, 33.6898270, -80.3737183, 80.3987122
27: -38.4736633, 22.4056721, -38.6032372, 22.4192505, -60.8929138, 61.0089073
28: -31.7058678, 25.9937439, -31.7871914, 26.0138206, -57.7196884, 57.7809296
29: -38.9662781, 24.8605843, -39.0864105, 24.9371853, -63.9034653, 63.9469948
30: -37.5954323, 26.3835793, -37.6985474, 26.4299068, -64.0253372, 64.0821228
31: -36.1444473, 25.6001892, -36.2407608, 25.6321678, -61.7766151, 61.8409500
32: -33.2404938, 25.2419701, -33.2565498, 25.2934284, -58.5339203, 58.4985199
33: -54.8394318, 33.3147964, -54.9089432, 33.4120789, -88.2514954, 88.2237396
34: -47.4490318, 23.6585159, -47.5028152, 23.6813774, -71.1304092, 71.1613235
35: -43.3259659, 28.0843468, -43.3821182, 28.1176739, -71.4436417, 71.4664612
36: -41.4721527, 31.2366562, -41.5005798, 31.2698727, -72.7420197, 72.7372360
37: -60.9874039, 28.2105713, -61.0357323, 28.2524319, -89.2398224, 89.2463074
38: -53.1950531, 40.6984558, -53.2517242, 40.7453575, -93.9403992, 93.9501801
39: -62.2677650, 34.8016129, -62.3061447, 34.9147949, -97.1825562, 97.1077499
40: -54.9389572, 23.4455395, -54.9954376, 23.4766064, -78.4155579, 78.4409790
41: -35.5368423, 21.6976337, -35.5678024, 21.7322464, -57.2690887, 57.2654343
42: -29.7918434, 20.0195713, -29.8171272, 20.0891380, -49.8809814, 49.8367004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=478, inp2_unstable=479, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3072843, upper bound: 29.3288387
time: 73.23 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3072843, upper bound: 29.3435998
time: 84.02 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -46.4094086, 30.3720703, -46.4114151, 30.3121243, -76.7215347, 76.7834854
1: -27.7906246, 28.7630463, -27.7904091, 28.7317142, -56.5223389, 56.5534515
2: -24.6501160, 25.6300011, -24.6498260, 25.5643044, -50.2144165, 50.2798271
3: -30.7911129, 28.7326603, -30.7976589, 28.6436520, -59.4347610, 59.5303192
4: -29.2279110, 27.2875481, -29.2270889, 27.2327957, -56.4607086, 56.5146294
5: -30.3380547, 31.1986275, -30.3403435, 31.0975113, -61.4355621, 61.5389671
6: -36.4435425, 22.8478699, -36.4222412, 22.8411465, -59.2846832, 59.2701111
7: -35.6818275, 32.0580711, -35.6768456, 31.9976177, -67.6794434, 67.7349167
8: -38.5516891, 30.0523891, -38.5510330, 29.9512081, -68.5028992, 68.6034241
9: -31.9054279, 27.0232944, -31.8962021, 27.0353756, -58.9407959, 58.9194908
10: -43.9957428, 37.9041023, -43.9187622, 37.9144669, -81.9102097, 81.8228607
11: -42.6408043, 33.3785553, -42.6260071, 33.3776093, -76.0184174, 76.0045624
12: -38.0914688, 35.8285217, -37.8787842, 35.8273354, -73.9188080, 73.7072906
13: -40.1412048, 43.5311356, -40.1170197, 43.5446548, -83.6858597, 83.6481552
14: -66.1628723, 23.5152245, -66.1240387, 23.5061836, -89.6690521, 89.6392670
15: -36.8594818, 25.2329674, -36.8576698, 25.2060795, -62.0655594, 62.0906296
16: -45.9293900, 31.7136288, -45.9178009, 31.7156143, -77.6450043, 77.6314316
17: -66.0207520, 42.1806946, -66.0008240, 42.1808052, -108.2015381, 108.1815186
18: -37.9920845, 33.6407585, -37.9934425, 33.6352234, -71.6273041, 71.6342010
19: -31.9516582, 20.5589027, -31.9400482, 20.5588951, -52.5105515, 52.4989510
20: -27.3767681, 20.8881435, -27.3555145, 20.8842659, -48.2610321, 48.2436600
21: -38.4645119, 23.8265038, -38.4418297, 23.8152847, -62.2797966, 62.2683334
22: -37.5343628, 25.1408310, -37.5645790, 25.1301727, -62.6645279, 62.7054024
23: -31.5690727, 22.5316849, -31.5764771, 22.5216999, -54.0907745, 54.1081619
24: -32.5734329, 23.6700363, -32.6155319, 23.6652985, -56.2387238, 56.2855682
25: -29.9386501, 28.3151360, -29.9438705, 28.3040257, -58.2426758, 58.2590065
26: -46.9074821, 33.7562256, -46.8286743, 33.7479515, -80.6554260, 80.5848999
27: -38.5442810, 22.4245987, -38.6170044, 22.4214993, -60.9657822, 61.0415955
28: -31.7734756, 26.0300961, -31.7959175, 26.0234318, -57.7969055, 57.8260117
29: -39.0562820, 24.9861012, -39.0941887, 24.9798698, -64.0361481, 64.0802917
30: -37.6538086, 26.4634037, -37.7056732, 26.4528027, -64.1066132, 64.1690750
31: -36.2415314, 25.6524849, -36.2528763, 25.6467724, -61.8882904, 61.9053612
32: -33.3349075, 25.3175468, -33.2652817, 25.3174629, -58.6523705, 58.5828285
33: -54.9577255, 33.4153214, -54.9432678, 33.4228592, -88.3805847, 88.3585892
34: -47.5348434, 23.7243156, -47.5263062, 23.6892090, -71.2240524, 71.2506180
35: -43.4106178, 28.1457157, -43.4074860, 28.1244392, -71.5350494, 71.5531921
36: -41.5274506, 31.2804394, -41.5094376, 31.2807598, -72.8082123, 72.7898788
37: -61.0693474, 28.2843666, -61.0483742, 28.2721100, -89.3414612, 89.3327332
38: -53.2865791, 40.7618179, -53.2700577, 40.7608833, -94.0474548, 94.0318756
39: -62.3497238, 34.8553581, -62.3233986, 34.9221649, -97.2718811, 97.1787491
40: -55.0261116, 23.4750423, -55.0143433, 23.4792843, -78.5053940, 78.4893799
41: -35.5979805, 21.7449913, -35.5785828, 21.7419586, -57.3399353, 57.3235664
42: -29.8574600, 20.1254978, -29.8254147, 20.1215820, -49.9790344, 49.9509087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=478, inp2_unstable=479, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3435999, upper bound: 29.3288387
time: 89.80 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3435999, upper bound: 29.3435998
time: 71.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 163.94 seconds
IS_B2_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 163.94
Output dim: 2, lower bound: -29.2677246, upper bound: 29.2901069
IS_B2_A1_A2_B2, status: Status.VERIFIED, split count: 4, time: 163.94
Output dim: 2, lower bound: -29.2677247, upper bound: 29.3051341
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 163.94
Output dim: 2, lower bound: -29.3072843, upper bound: 29.3288387
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 163.94
Output dim: 2, lower bound: -29.3072843, upper bound: 29.3435998
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 163.94
Output dim: 2, lower bound: -29.3435999, upper bound: 29.3288387
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 163.94
Output dim: 2, lower bound: -29.3435999, upper bound: 29.3435998

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -46.2358208, 30.2286434, -46.2473106, 30.2265129, -76.4623260, 76.4759521
1: -27.6781807, 28.6740189, -27.6806984, 28.6587181, -56.3368912, 56.3547173
2: -24.4633808, 25.4953022, -24.4628143, 25.5188160, -49.9821968, 49.9581146
3: -30.6016693, 28.5631447, -30.6209240, 28.5752964, -59.1769562, 59.1840668
4: -29.0175686, 27.1590881, -28.9973030, 27.1664677, -56.1840363, 56.1563911
5: -30.1498642, 31.0090084, -30.1671619, 31.0256386, -61.1755028, 61.1761627
6: -36.3349915, 22.7383614, -36.3285637, 22.7019424, -59.0369339, 59.0669250
7: -35.5102005, 31.9364491, -35.5285034, 31.9135780, -67.4237823, 67.4649506
8: -38.3255692, 29.9081707, -38.3023148, 29.8799133, -68.2054825, 68.2104874
9: -31.8369522, 26.8338642, -31.8285942, 26.8163528, -58.6533051, 58.6624527
10: -43.8226929, 37.5048752, -43.8098831, 37.4163704, -81.2390594, 81.3147583
11: -42.4584770, 33.0722427, -42.5381813, 33.0037918, -75.4622650, 75.6104202
12: -37.7875443, 35.3886833, -37.7944031, 35.3230972, -73.1106415, 73.1830902
13: -40.0734024, 43.3608208, -40.0517349, 43.3327904, -83.4061890, 83.4125519
14: -65.9237671, 23.2025375, -65.9893188, 23.1123180, -89.0360718, 89.1918488
15: -36.6709862, 25.1294041, -36.6340141, 25.1253128, -61.7962952, 61.7634201
16: -45.8021545, 31.4840736, -45.8047104, 31.4190445, -77.2211914, 77.2887802
17: -65.7838593, 41.7814636, -65.9124908, 41.6968651, -107.4807129, 107.6939545
18: -37.8189201, 33.4803467, -37.8641930, 33.4867935, -71.3057098, 71.3445435
19: -31.8205185, 20.4743843, -31.8555374, 20.4855671, -52.3060837, 52.3299217
20: -27.2479782, 20.7836246, -27.2801056, 20.7726078, -48.0205841, 48.0637283
21: -38.2924309, 23.6492500, -38.3566704, 23.6355915, -61.9280243, 62.0059204
22: -37.3788872, 24.9781742, -37.4128265, 24.9847755, -62.3636627, 62.3909988
23: -31.4645615, 22.4409370, -31.4944134, 22.4319611, -53.8965225, 53.9353447
24: -32.4311028, 23.6311340, -32.4241409, 23.6282253, -56.0593262, 56.0552750
25: -29.8367844, 28.2360611, -29.8316784, 28.2373562, -58.0741425, 58.0677414
26: -46.6402435, 33.4809647, -46.7298012, 33.4906921, -80.1309357, 80.2107697
27: -38.3652191, 22.3850555, -38.3816833, 22.3778820, -60.7431030, 60.7667389
28: -31.6544914, 25.9723530, -31.6815796, 25.9701786, -57.6246643, 57.6539307
29: -38.9189072, 24.7819481, -38.9902382, 24.7763615, -63.6952667, 63.7721863
30: -37.5486565, 26.3230000, -37.6028023, 26.3062382, -63.8548889, 63.9258003
31: -36.0927353, 25.5698318, -36.1353378, 25.5710144, -61.6637421, 61.7051697
32: -33.2000961, 25.1625557, -33.1735992, 25.1307220, -58.3308105, 58.3361549
33: -54.7048340, 33.2868118, -54.6326027, 33.3547707, -88.0596008, 87.9194183
34: -47.3513870, 23.6341629, -47.3029251, 23.6317368, -70.9831085, 70.9370880
35: -43.2099762, 28.0631313, -43.1446915, 28.0741673, -71.2841415, 71.2078247
36: -41.3989105, 31.2167168, -41.3509827, 31.2285404, -72.6274490, 72.5676956
37: -60.8841019, 28.1830997, -60.8244095, 28.1954384, -89.0795288, 89.0075073
38: -53.1048317, 40.6620407, -53.0679436, 40.6720352, -93.7768555, 93.7299805
39: -62.1623192, 34.7823639, -62.0933495, 34.8753662, -97.0376740, 96.8757172
40: -54.8297157, 23.4266968, -54.7705574, 23.4386959, -78.2684021, 78.1972504
41: -35.4798050, 21.6662483, -35.4512558, 21.6686935, -57.1484833, 57.1174965
42: -29.7564468, 19.9107265, -29.7443390, 19.8719578, -49.6284027, 49.6550636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_B2_A2_A1_B1_B1

### Relational analysis result of IS_B2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.3050259, upper bound: 29.2760652
time: 82.27 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2

### Relational analysis result of IS_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3050259, upper bound: 29.3261317
time: 154.08 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -46.2868385, 30.2558289, -46.3993721, 30.3234634, -76.6102905, 76.6551971
1: -27.7164078, 28.6992874, -27.7898884, 28.7415733, -56.4579773, 56.4891739
2: -24.5260735, 25.5102139, -24.6228676, 25.5886021, -50.1146774, 50.1330795
3: -30.6496201, 28.5874004, -30.7530689, 28.6692047, -59.3188171, 59.3404655
4: -29.0900898, 27.1811180, -29.1897984, 27.2786865, -56.3687744, 56.3709183
5: -30.1976376, 31.0349274, -30.3002548, 31.1279068, -61.3255463, 61.3351784
6: -36.3670044, 22.7810154, -36.4562531, 22.8255634, -59.1925659, 59.2372665
7: -35.5550232, 31.9621601, -35.6775894, 31.9967804, -67.5518036, 67.6397476
8: -38.4124832, 29.9339695, -38.5164490, 29.9981155, -68.4105988, 68.4504166
9: -31.8599415, 26.9176331, -31.9547405, 27.0192413, -58.8791656, 58.8723717
10: -43.8611984, 37.6914520, -44.1270447, 37.8440857, -81.7052841, 81.8184967
11: -42.4901924, 33.2070618, -42.7885895, 33.3134727, -75.8036652, 75.9956436
12: -37.8153152, 35.5698891, -38.1206017, 35.7404900, -73.5558014, 73.6904907
13: -40.0933571, 43.4443970, -40.1651993, 43.5588684, -83.6522217, 83.6095886
14: -65.9664612, 23.3523369, -66.3197174, 23.4503841, -89.4168396, 89.6720505
15: -36.7543716, 25.1585274, -36.8546753, 25.2554340, -62.0098038, 62.0131989
16: -45.8427544, 31.5955811, -45.9902573, 31.6866322, -77.5293808, 77.5858383
17: -65.8110352, 41.9605789, -66.1922913, 42.1117172, -107.9227524, 108.1528702
18: -37.8588829, 33.5194054, -38.0250587, 33.6006241, -71.4595032, 71.5444641
19: -31.8470306, 20.4930305, -31.9645309, 20.5465584, -52.3935852, 52.4575577
20: -27.2746658, 20.8215675, -27.4154358, 20.8677216, -48.1423874, 48.2369995
21: -38.3222618, 23.7091026, -38.5312996, 23.7877083, -62.1099701, 62.2404022
22: -37.4250488, 25.0219994, -37.5624962, 25.1217308, -62.5467758, 62.5844955
23: -31.4941120, 22.4703293, -31.6006184, 22.5156364, -54.0097504, 54.0709457
24: -32.5069885, 23.6417084, -32.6168861, 23.7154408, -56.2224274, 56.2585945
25: -29.8775043, 28.2578049, -29.9514828, 28.3267059, -58.2042084, 58.2092781
26: -46.6742477, 33.5616150, -46.8986549, 33.6925964, -80.3668442, 80.4602661
27: -38.4594345, 22.3982544, -38.6221695, 22.4821453, -60.9415741, 61.0204201
28: -31.6982174, 25.9866562, -31.8038807, 26.0501213, -57.7483368, 57.7905350
29: -38.9540291, 24.8466167, -39.1136780, 24.9384136, -63.8924408, 63.9602852
30: -37.5874825, 26.3716087, -37.7277260, 26.4373989, -64.0248795, 64.0993347
31: -36.1295433, 25.5903378, -36.2789001, 25.6380539, -61.7675972, 61.8692322
32: -33.2318649, 25.2302475, -33.3290558, 25.2918816, -58.5237427, 58.5593033
33: -54.8198166, 33.3092003, -54.9140816, 33.5376205, -88.3574371, 88.2232819
34: -47.4323959, 23.6534996, -47.5095367, 23.7935791, -71.2259674, 71.1630325
35: -43.3078690, 28.0808811, -43.3785553, 28.2483597, -71.5562210, 71.4594269
36: -41.4580841, 31.2337284, -41.5101471, 31.3054466, -72.7635345, 72.7438736
37: -60.9681168, 28.2052765, -61.0571365, 28.3015766, -89.2696915, 89.2624130
38: -53.1744194, 40.6880836, -53.2692337, 40.7711220, -93.9455414, 93.9573212
39: -62.2465019, 34.7973633, -62.3196220, 34.9601555, -97.2066574, 97.1169891
40: -54.9239159, 23.4392948, -55.0296402, 23.5152645, -78.4391785, 78.4689331
41: -35.5267792, 21.6895733, -35.5898666, 21.7441387, -57.2709198, 57.2794418
42: -29.7847309, 19.9968414, -29.8731422, 20.0814781, -49.8662109, 49.8699799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2918265, upper bound: 29.3409120
time: 75.73 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.3050258, upper bound: 29.3022779
time: 121.08 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -46.3437843, 30.3330460, -46.2784767, 30.2324696, -76.5762558, 76.6115189
1: -27.7454567, 28.7294807, -27.7000008, 28.6631470, -56.4086037, 56.4294815
2: -24.5768242, 25.6096249, -24.5010834, 25.5228806, -50.0997009, 50.1107063
3: -30.7241344, 28.7022705, -30.6623707, 28.5815697, -59.3057022, 59.3646393
4: -29.1342030, 27.2574196, -29.0360432, 27.1717339, -56.3059387, 56.2934570
5: -30.2729244, 31.1664448, -30.2081184, 31.0316391, -61.3045654, 61.3745613
6: -36.4023094, 22.7828579, -36.3380280, 22.7090397, -59.1113434, 59.1208878
7: -35.6245422, 32.0191460, -35.5602341, 31.9185028, -67.5430450, 67.5793762
8: -38.4500961, 30.0202885, -38.3430405, 29.8855038, -68.3356018, 68.3633270
9: -31.8765392, 26.9259644, -31.8372726, 26.8386803, -58.7152100, 58.7632370
10: -43.9489861, 37.6887512, -43.8226357, 37.4726334, -81.4216156, 81.5113831
11: -42.6029091, 33.2206116, -42.5481033, 33.0541039, -75.6570129, 75.7687149
12: -38.0537033, 35.6217041, -37.8022156, 35.4030075, -73.4566956, 73.4239197
13: -40.1099472, 43.4348602, -40.0542145, 43.3468437, -83.4567871, 83.4890747
14: -66.1058731, 23.3452892, -66.0086823, 23.1599998, -89.2658691, 89.3539734
15: -36.7582474, 25.1977558, -36.6529884, 25.1340771, -61.8923187, 61.8507385
16: -45.8788528, 31.5793877, -45.8150253, 31.4424038, -77.3212509, 77.3944092
17: -65.9831085, 41.9759254, -65.9254150, 41.7630196, -107.7461243, 107.9013367
18: -37.9340019, 33.5830383, -37.8752708, 33.5194588, -71.4534607, 71.4583130
19: -31.9146538, 20.5314045, -31.8644123, 20.5035934, -52.4182472, 52.3958168
20: -27.3448715, 20.8422756, -27.2898998, 20.7905312, -48.1354027, 48.1321754
21: -38.4274178, 23.7548275, -38.3659210, 23.6690388, -62.0964508, 62.1207390
22: -37.4641418, 25.0840302, -37.4217834, 25.0145760, -62.4787140, 62.5058136
23: -31.5328884, 22.4949913, -31.5021935, 22.4472809, -53.9801636, 53.9971809
24: -32.4845657, 23.6536827, -32.4337540, 23.6323357, -56.1168976, 56.0874367
25: -29.8867111, 28.2880363, -29.8381824, 28.2489185, -58.1356277, 58.1262207
26: -46.8639069, 33.6578598, -46.7395287, 33.5483704, -80.4122620, 80.3973846
27: -38.4358940, 22.4039364, -38.3954048, 22.3800240, -60.8159180, 60.7993393
28: -31.7222099, 26.0086422, -31.6903229, 25.9798260, -57.7020264, 57.6989670
29: -39.0088310, 24.9072609, -38.9980164, 24.8189640, -63.8277969, 63.9052773
30: -37.6070251, 26.4024391, -37.6099091, 26.3287849, -63.9358101, 64.0123444
31: -36.1900787, 25.6220551, -36.1472626, 25.5856152, -61.7756958, 61.7693176
32: -33.2945404, 25.2380161, -33.1823730, 25.1547375, -58.4492798, 58.4203873
33: -54.8230476, 33.3874893, -54.6669312, 33.3656845, -88.1887207, 88.0544205
34: -47.4371109, 23.7000217, -47.3263512, 23.6396618, -71.0767746, 71.0263672
35: -43.2945175, 28.1245575, -43.1700401, 28.0809879, -71.3755035, 71.2946014
36: -41.4541245, 31.2599106, -41.3598785, 31.2387218, -72.6928482, 72.6197891
37: -60.9659729, 28.2561378, -60.8370476, 28.2149124, -89.1808853, 89.0931854
38: -53.1961288, 40.7253456, -53.0862465, 40.6876869, -93.8838196, 93.8115921
39: -62.2441521, 34.8360558, -62.1104393, 34.8827362, -97.1268921, 96.9464951
40: -54.9167709, 23.4561443, -54.7893295, 23.4413757, -78.3581467, 78.2454681
41: -35.5414047, 21.7135715, -35.4619942, 21.6784973, -57.2199020, 57.1755676
42: -29.8220329, 20.0164986, -29.7526836, 19.9042053, -49.7262383, 49.7691803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=478, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1688

## Relational analysis of IS_B2_A2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3398220, upper bound: 29.3018146
time: 89.88 seconds

## Relational analysis of IS_B2_A2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2617198, upper bound: 29.2874247
time: 84.42 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -46.3950233, 30.3605003, -46.4307289, 30.3294182, -76.7244415, 76.7912216
1: -27.7839813, 28.7547531, -27.8093033, 28.7460022, -56.5299835, 56.5640526
2: -24.6396027, 25.6245174, -24.6612110, 25.5926266, -50.2322235, 50.2857285
3: -30.7722740, 28.7264500, -30.7947159, 28.6755180, -59.4477844, 59.5211601
4: -29.2072029, 27.2795067, -29.2288990, 27.2840214, -56.4912262, 56.5084076
5: -30.3209133, 31.1922760, -30.3414593, 31.1337986, -61.4547119, 61.5337334
6: -36.4343987, 22.8254242, -36.4658356, 22.8325253, -59.2669220, 59.2912560
7: -35.6697273, 32.0448380, -35.7097054, 32.0017815, -67.6715088, 67.7545395
8: -38.5372009, 30.0460415, -38.5573044, 30.0036392, -68.5408325, 68.6033478
9: -31.8994770, 27.0099125, -31.9633656, 27.0416489, -58.9411240, 58.9732742
10: -43.9872055, 37.8754196, -44.1397324, 37.9004097, -81.8876190, 82.0151443
11: -42.6344719, 33.3555222, -42.7984161, 33.3638763, -75.9983521, 76.1539307
12: -38.0813713, 35.8030701, -38.1283951, 35.8204956, -73.9018707, 73.9314575
13: -40.1298714, 43.5184784, -40.1675873, 43.5731354, -83.7030029, 83.6860580
14: -66.1484070, 23.4951973, -66.3390045, 23.4982414, -89.6466522, 89.8341980
15: -36.8422394, 25.2269287, -36.8739853, 25.2642365, -62.1064606, 62.1009102
16: -45.9195862, 31.6911201, -46.0004463, 31.7101898, -77.6297684, 77.6915588
17: -66.0102386, 42.1552238, -66.2050781, 42.1779938, -108.1882324, 108.3603058
18: -37.9735413, 33.6221619, -38.0365753, 33.6334305, -71.6069641, 71.6587372
19: -31.9411469, 20.5500832, -31.9734669, 20.5646629, -52.5057983, 52.5235443
20: -27.3713913, 20.8802872, -27.4250793, 20.8856907, -48.2570801, 48.3053665
21: -38.4572334, 23.8147583, -38.5405045, 23.8211670, -62.2783966, 62.3552628
22: -37.5104027, 25.1280861, -37.5715561, 25.1517735, -62.6621780, 62.6996422
23: -31.5624447, 22.5244522, -31.6084709, 22.5309563, -54.0933990, 54.1329231
24: -32.5607147, 23.6642876, -32.6266785, 23.7194023, -56.2801170, 56.2909622
25: -29.9274826, 28.3100395, -29.9582081, 28.3383160, -58.2657852, 58.2682419
26: -46.8978729, 33.7373505, -46.9083786, 33.7506180, -80.6484909, 80.6457291
27: -38.5300102, 22.4171371, -38.6360092, 22.4843655, -61.0143738, 61.0531464
28: -31.7657986, 26.0230331, -31.8126526, 26.0597591, -57.8255577, 57.8356819
29: -39.0439720, 24.9721565, -39.1214981, 24.9811268, -64.0251007, 64.0936584
30: -37.6458206, 26.4514370, -37.7348709, 26.4602833, -64.1061020, 64.1863098
31: -36.2264175, 25.6425972, -36.2911301, 25.6527500, -61.8791656, 61.9337234
32: -33.3262711, 25.3058205, -33.3378220, 25.3159370, -58.6421967, 58.6436424
33: -54.9380379, 33.4097290, -54.9484634, 33.5483932, -88.4864349, 88.3581924
34: -47.5182114, 23.7192860, -47.5331116, 23.8014088, -71.3196106, 71.2523956
35: -43.3925018, 28.1422291, -43.4039192, 28.2551441, -71.6476440, 71.5461426
36: -41.5133286, 31.2774143, -41.5190506, 31.3159180, -72.8292389, 72.7964630
37: -61.0491867, 28.2788715, -61.0699234, 28.3212509, -89.3704376, 89.3487930
38: -53.2659683, 40.7512436, -53.2877121, 40.7866669, -94.0526276, 94.0389557
39: -62.3284950, 34.8510399, -62.3374786, 34.9675293, -97.2960205, 97.1885223
40: -55.0109978, 23.4687729, -55.0485344, 23.5179157, -78.5289154, 78.5173035
41: -35.5877304, 21.7369118, -35.6007843, 21.7538986, -57.3416138, 57.3376923
42: -29.8503494, 20.1030483, -29.8815403, 20.1143284, -49.9646759, 49.9845886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=478, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_B2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3176420, upper bound: 29.3398219
time: 78.08 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3398219, upper bound: 29.3398219
time: 87.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 168.20 seconds
IS_B2_A2_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 168.20
Output dim: 2, lower bound: -29.3050259, upper bound: 29.2760652
IS_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 168.20
Output dim: 2, lower bound: -29.3050259, upper bound: 29.3261317
IS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 168.20
Output dim: 2, lower bound: -29.2918265, upper bound: 29.3409120
IS_B2_A2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 168.20
Output dim: 2, lower bound: -29.3050258, upper bound: 29.3022779
IS_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 168.20
Output dim: 2, lower bound: -29.3398220, upper bound: 29.3018146
IS_B2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 168.20
Output dim: 2, lower bound: -29.2617198, upper bound: 29.2874247
IS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 168.20
Output dim: 2, lower bound: -29.3176420, upper bound: 29.3398219
IS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 168.20
Output dim: 2, lower bound: -29.3398219, upper bound: 29.3398219

## BFS IS instance: IS_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -46.2207985, 30.2223034, -46.2707787, 30.2643204, -76.4851227, 76.4930801
1: -27.6678867, 28.6655693, -27.7039413, 28.6781330, -56.3460083, 56.3695107
2: -24.4504585, 25.4901543, -24.4737930, 25.5690651, -50.0195160, 49.9639435
3: -30.5872040, 28.5560455, -30.6254959, 28.6623955, -59.2495995, 59.1815414
4: -28.9995079, 27.1534920, -29.0008621, 27.2708511, -56.2703590, 56.1543503
5: -30.1355934, 31.0018711, -30.1693382, 31.1319313, -61.2675247, 61.1712036
6: -36.3256950, 22.7204151, -36.3694077, 22.7116356, -59.0373230, 59.0898209
7: -35.4922447, 31.9190273, -35.5523796, 31.9131050, -67.4053497, 67.4714050
8: -38.3097000, 29.9003334, -38.3163528, 29.9513130, -68.2610168, 68.2166901
9: -31.8276367, 26.8167992, -31.8402348, 26.8457546, -58.6733856, 58.6570358
10: -43.8130417, 37.4822006, -43.8646698, 37.4493446, -81.2623749, 81.3468628
11: -42.4489899, 33.0434914, -42.7359619, 32.9910049, -75.4399872, 75.7794495
12: -37.7767372, 35.3598557, -38.0463638, 35.3151741, -73.0919113, 73.4062195
13: -40.0630493, 43.3416519, -40.1010361, 43.3514481, -83.4144897, 83.4426880
14: -65.9071045, 23.1850910, -66.1852264, 23.1092148, -89.0163193, 89.3703156
15: -36.6455002, 25.1218700, -36.6349983, 25.2446060, -61.8901062, 61.7568626
16: -45.7890434, 31.4581699, -45.8654404, 31.4370384, -77.2260742, 77.3236084
17: -65.7703552, 41.7462044, -66.2057724, 41.6731491, -107.4435043, 107.9519730
18: -37.7955132, 33.4674568, -37.8841248, 33.5056229, -71.3011169, 71.3515778
19: -31.8137760, 20.4624100, -31.9743080, 20.4883041, -52.3020782, 52.4367180
20: -27.2403316, 20.7713375, -27.3670883, 20.7766685, -48.0169983, 48.1384201
21: -38.2845764, 23.6328278, -38.5211372, 23.6409893, -61.9255676, 62.1539612
22: -37.3585358, 24.9654121, -37.4391022, 25.0079651, -62.3665009, 62.4045029
23: -31.4583282, 22.4299412, -31.5661449, 22.4385548, -53.8968811, 53.9960861
24: -32.4166946, 23.6269741, -32.4468307, 23.6562653, -56.0729599, 56.0738029
25: -29.8238220, 28.2280407, -29.8686104, 28.2576733, -58.0814972, 58.0966492
26: -46.6272850, 33.4592705, -46.8242722, 33.4962845, -80.1235657, 80.2835388
27: -38.3435974, 22.3780537, -38.3897285, 22.3901272, -60.7337265, 60.7677841
28: -31.6483536, 25.9623623, -31.7453861, 25.9714603, -57.6198120, 57.7077446
29: -38.9045143, 24.7645493, -39.0640144, 24.7751827, -63.6796951, 63.8285637
30: -37.5395508, 26.3017769, -37.6493034, 26.3059082, -63.8454590, 63.9510803
31: -36.0844231, 25.5594826, -36.2184677, 25.5762939, -61.6607170, 61.7779465
32: -33.1899643, 25.1467342, -33.2519951, 25.1345062, -58.3244705, 58.3987274
33: -54.6831512, 33.2770348, -54.6651382, 33.4202156, -88.1033630, 87.9421692
34: -47.3346367, 23.6266022, -47.3166885, 23.7583961, -71.0930328, 70.9432907
35: -43.1921997, 28.0564041, -43.1649475, 28.1257668, -71.3179626, 71.2213516
36: -41.3808060, 31.2076836, -41.3726044, 31.2429256, -72.6237335, 72.5802917
37: -60.8649521, 28.1720219, -60.8708191, 28.2230396, -89.0879822, 89.0428391
38: -53.0788116, 40.6500626, -53.1000328, 40.6974335, -93.7762451, 93.7500916
39: -62.1415443, 34.7658348, -62.1406555, 34.8934097, -97.0349426, 96.9064865
40: -54.8125038, 23.4167290, -54.7966309, 23.5086613, -78.3211594, 78.2133636
41: -35.4657211, 21.6525612, -35.4736824, 21.6792412, -57.1449585, 57.1262436
42: -29.7497673, 19.8915749, -29.7980175, 19.8713264, -49.6210938, 49.6895905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=255, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=476, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1690

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_B2_A2_A1_B1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2892864, upper bound: 29.3230069
time: 78.88 seconds

## Relational analysis of IS_B2_A2_A1_B1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2455131, upper bound: 29.2853131
time: 75.88 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -46.1586494, 30.2072372, -46.3406219, 30.3015671, -76.4602127, 76.5478592
1: -27.6401577, 28.6505775, -27.7554588, 28.7195110, -56.3596687, 56.4060364
2: -24.4110584, 25.4752769, -24.5696163, 25.5729923, -49.9840508, 50.0448914
3: -30.5144234, 28.5414314, -30.6907864, 28.6484013, -59.1628265, 59.2322159
4: -28.9239941, 27.1453953, -29.1138744, 27.2626820, -56.1866760, 56.2592697
5: -30.0697937, 30.9810543, -30.2422199, 31.1037636, -61.1735573, 61.2232742
6: -36.3120346, 22.6427155, -36.4312363, 22.7616749, -59.0737076, 59.0739517
7: -35.4763718, 31.8958969, -35.6417694, 31.9663925, -67.4427643, 67.5376663
8: -38.2567062, 29.8842182, -38.4451408, 29.9758568, -68.2325592, 68.3293610
9: -31.7829914, 26.8291187, -31.9198666, 26.9785080, -58.7614899, 58.7489853
10: -43.7877235, 37.5000305, -44.0939865, 37.7565193, -81.5442352, 81.5940094
11: -42.4218712, 32.9140015, -42.7578278, 33.1804123, -75.6022797, 75.6718216
12: -37.7449341, 35.2627869, -38.0891495, 35.6006432, -73.3455811, 73.3519363
13: -40.0384521, 43.2883453, -40.1405869, 43.4883423, -83.5267944, 83.4289322
14: -65.8760986, 23.1069717, -66.2787476, 23.3383942, -89.2144928, 89.3857193
15: -36.5305557, 25.1117115, -36.7528152, 25.2340279, -61.7645836, 61.8645248
16: -45.7647171, 31.3967476, -45.9550056, 31.5961304, -77.3608398, 77.3517532
17: -65.7304993, 41.5639191, -66.1559906, 41.9328842, -107.6633606, 107.7199020
18: -37.7669220, 33.4357910, -37.9825592, 33.5622330, -71.3291473, 71.4183502
19: -31.7958031, 20.3687706, -31.9413090, 20.4904785, -52.2862778, 52.3100777
20: -27.2270279, 20.6990051, -27.3941135, 20.8125038, -48.0395317, 48.0931168
21: -38.2671471, 23.5252342, -38.5065079, 23.7049046, -61.9720535, 62.0317383
22: -37.3293610, 24.8843079, -37.5186653, 25.0596409, -62.3889999, 62.4029694
23: -31.4479713, 22.3678226, -31.5796146, 22.4691372, -53.9170990, 53.9474373
24: -32.4146080, 23.6131115, -32.5746422, 23.7025719, -56.1171722, 56.1877518
25: -29.8021107, 28.1727390, -29.9171085, 28.2883701, -58.0904770, 58.0898438
26: -46.5994415, 33.4039040, -46.8647881, 33.6215553, -80.2209930, 80.2686920
27: -38.3580856, 22.3599586, -38.5759926, 22.4650288, -60.8231125, 60.9359512
28: -31.6537457, 25.8990803, -31.7836723, 26.0104866, -57.6642303, 57.6827507
29: -38.8979683, 24.6597748, -39.0882568, 24.8539143, -63.7518768, 63.7480316
30: -37.5328674, 26.2438240, -37.7029839, 26.3790226, -63.9118881, 63.9468079
31: -36.0680313, 25.4888821, -36.2510300, 25.5924568, -61.6604843, 61.7399139
32: -33.1799469, 25.0919514, -33.3054886, 25.2283859, -58.4083328, 58.3974380
33: -54.6183052, 33.2581978, -54.8228226, 33.5144043, -88.1327057, 88.0810242
34: -47.2558823, 23.6098690, -47.4302788, 23.7738457, -71.0297241, 71.0401459
35: -43.1663475, 28.0479698, -43.3146629, 28.2334366, -71.3997803, 71.3626251
36: -41.3888702, 31.1628284, -41.4787979, 31.2734814, -72.6623459, 72.6416245
37: -60.8301849, 28.1490135, -60.9943848, 28.2756310, -89.1058197, 89.1434021
38: -53.0711250, 40.6216354, -53.2214317, 40.7409363, -93.8120575, 93.8430634
39: -62.0923386, 34.7547493, -62.2495880, 34.9406433, -97.0329819, 97.0043335
40: -54.7402000, 23.3929558, -54.9473801, 23.4941711, -78.2343750, 78.3403320
41: -35.4472809, 21.5996113, -35.5537758, 21.7025967, -57.1498795, 57.1533890
42: -29.7278328, 19.8383408, -29.8472633, 20.0088615, -49.7366943, 49.6856041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=476, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_B2_A2_A1_B2_A1_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2749319, upper bound: 29.3377528
time: 70.69 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2858441, upper bound: 29.3377528
time: 68.10 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -46.2952805, 30.3151779, -46.1716766, 30.1928215, -76.4880981, 76.4868546
1: -27.7147808, 28.7131920, -27.6330833, 28.6270638, -56.3418427, 56.3462753
2: -24.5184898, 25.5985641, -24.3734474, 25.4984398, -50.0169220, 49.9720116
3: -30.6689434, 28.6842594, -30.5399303, 28.5413513, -59.2102966, 59.2241898
4: -29.0655270, 27.2421627, -28.8858471, 27.1384926, -56.2040024, 56.1280060
5: -30.2166653, 31.1472702, -30.0838547, 30.9889297, -61.2055969, 61.2311249
6: -36.3808746, 22.7348270, -36.2905197, 22.6040478, -58.9849243, 59.0253448
7: -35.5783157, 32.0021362, -35.4580078, 31.8813934, -67.4597092, 67.4601364
8: -38.3801498, 30.0031357, -38.1876526, 29.8471508, -68.2272949, 68.1907806
9: -31.8573322, 26.8694630, -31.7944832, 26.7142715, -58.5716019, 58.6639366
10: -43.9220695, 37.5764008, -43.7622032, 37.2235222, -81.1455917, 81.3386078
11: -42.5790253, 33.1270294, -42.4945793, 32.8464470, -75.4254761, 75.6216049
12: -38.0305939, 35.4716721, -37.7512436, 35.0684547, -73.0990448, 73.2229156
13: -40.0903854, 43.3702469, -40.0119362, 43.2038307, -83.2942200, 83.3821869
14: -66.0672836, 23.2425957, -65.9238434, 22.9319839, -88.9992676, 89.1664276
15: -36.6873093, 25.1779213, -36.5004654, 25.0901203, -61.7774200, 61.6783791
16: -45.8465271, 31.5073948, -45.7437820, 31.2876587, -77.1341858, 77.2511749
17: -65.9554214, 41.8341141, -65.8646698, 41.4455147, -107.4009399, 107.6987839
18: -37.8987274, 33.5468521, -37.7977371, 33.4405365, -71.3392563, 71.3445892
19: -31.8932934, 20.5085144, -31.8169518, 20.4537735, -52.3470688, 52.3254623
20: -27.3247585, 20.8098068, -27.2451553, 20.7183971, -48.0431519, 48.0549622
21: -38.4050102, 23.7054768, -38.3160057, 23.5591927, -61.9642029, 62.0214844
22: -37.4298172, 25.0376434, -37.3468475, 24.9129677, -62.3427734, 62.3844910
23: -31.5106792, 22.4738884, -31.4523220, 22.4018650, -53.9125443, 53.9262047
24: -32.4303589, 23.6438980, -32.3142700, 23.6111832, -56.0415344, 55.9581680
25: -29.8533173, 28.2703590, -29.7646084, 28.2097855, -58.0631027, 58.0349655
26: -46.8367424, 33.5815353, -46.6790466, 33.3829842, -80.2197113, 80.2605820
27: -38.3786659, 22.3934116, -38.2697678, 22.3575611, -60.7362213, 60.6631775
28: -31.7001743, 25.9966087, -31.6413002, 25.9534512, -57.6536255, 57.6379089
29: -38.9858437, 24.8438778, -38.9478378, 24.6791534, -63.6649971, 63.7917137
30: -37.5806885, 26.3664265, -37.5517273, 26.2502460, -63.8309326, 63.9181519
31: -36.1581421, 25.6048183, -36.0776138, 25.5485649, -61.7067070, 61.6824303
32: -33.2732849, 25.1735458, -33.1351166, 25.0113163, -58.2845993, 58.3086586
33: -54.7424774, 33.3677673, -54.4880867, 33.3216095, -88.0640869, 87.8558502
34: -47.3891907, 23.6845169, -47.2206268, 23.6052532, -70.9944458, 70.9051437
35: -43.2387161, 28.1120300, -43.0472641, 28.0529575, -71.2916718, 71.1592941
36: -41.4306488, 31.2332649, -41.3088989, 31.1793060, -72.6099548, 72.5421600
37: -60.9170837, 28.2348461, -60.7289963, 28.1689682, -89.0860443, 88.9638367
38: -53.1568718, 40.6834335, -52.9997292, 40.5989189, -93.7557907, 93.6831589
39: -62.1930542, 34.8211136, -61.9996490, 34.8498917, -97.0429382, 96.8207550
40: -54.8555527, 23.4426651, -54.6528206, 23.4125881, -78.2681427, 78.0954742
41: -35.5149002, 21.6816788, -35.4033966, 21.6086159, -57.1235046, 57.0850754
42: -29.8033371, 19.9468403, -29.7108250, 19.7509766, -49.5543137, 49.6576614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=255, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_B2_A2_A2_B1_B1_B1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3377796, upper bound: 29.2872267
time: 69.56 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B2

### Relational analysis result of IS_B2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3377796, upper bound: 29.2993193
time: 71.00 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -46.2863274, 30.3206902, -46.3820381, 30.3116760, -76.5979996, 76.7027283
1: -27.7152920, 28.7184258, -27.7788982, 28.7297668, -56.4450607, 56.4973221
2: -24.5094604, 25.6000729, -24.6024399, 25.5817928, -50.0912476, 50.2025146
3: -30.6477070, 28.6863136, -30.7389679, 28.6575241, -59.3052292, 59.4252815
4: -29.0512295, 27.2458229, -29.1591854, 27.2689495, -56.3201752, 56.4050064
5: -30.1961956, 31.1495972, -30.2856522, 31.1149235, -61.3111115, 61.4352493
6: -36.3861580, 22.7178402, -36.4441376, 22.7844353, -59.1705856, 59.1619759
7: -35.5653877, 32.0073166, -35.6631165, 31.9848804, -67.5502625, 67.6704254
8: -38.3785324, 30.0077362, -38.4866409, 29.9867439, -68.3652649, 68.4943771
9: -31.8565941, 26.8822060, -31.9443779, 26.9849796, -58.8415756, 58.8265839
10: -43.9268074, 37.6215019, -44.1128082, 37.7872429, -81.7140503, 81.7343140
11: -42.5807419, 33.1442833, -42.7746620, 33.2699242, -75.8506470, 75.9189453
12: -38.0302658, 35.4652863, -38.1057930, 35.6698952, -73.7001648, 73.5710754
13: -40.0869713, 43.3755341, -40.1486588, 43.5091553, -83.5961304, 83.5241852
14: -66.0629883, 23.2663193, -66.3007965, 23.3954029, -89.4583893, 89.5671158
15: -36.6831055, 25.1824245, -36.8036499, 25.2441845, -61.9272919, 61.9860687
16: -45.8476410, 31.5294456, -45.9687271, 31.6381454, -77.4857864, 77.4981689
17: -65.9488144, 41.8358917, -66.1777954, 42.0357819, -107.9845963, 108.0136871
18: -37.8944550, 33.5416985, -38.0005417, 33.5970001, -71.4914551, 71.5422363
19: -31.8933907, 20.4990883, -31.9520187, 20.5419540, -52.4353447, 52.4511070
20: -27.3264580, 20.8074780, -27.4052410, 20.8533287, -48.1797791, 48.2127190
21: -38.4069901, 23.7038651, -38.5182419, 23.7720413, -62.1790314, 62.2221069
22: -37.4332047, 25.0243340, -37.5370674, 25.1051941, -62.5383987, 62.5614014
23: -31.5126228, 22.4767551, -31.5858135, 22.5098248, -54.0224457, 54.0625648
24: -32.4382553, 23.6427040, -32.5723419, 23.7098618, -56.1481018, 56.2150459
25: -29.8525047, 28.2704830, -29.9246120, 28.3208218, -58.1733246, 58.1950912
26: -46.8370171, 33.5685654, -46.8813362, 33.6750488, -80.5120544, 80.4499054
27: -38.4008408, 22.3942337, -38.5788269, 22.4743595, -60.8751907, 60.9730530
28: -31.7165775, 25.9960213, -31.7902412, 26.0478344, -57.7644119, 57.7862625
29: -38.9926567, 24.8307686, -39.0985985, 24.9173775, -63.9100342, 63.9293671
30: -37.5867310, 26.3692455, -37.7085381, 26.4233704, -64.0100784, 64.0777817
31: -36.1551781, 25.6046371, -36.2579422, 25.6358833, -61.7910614, 61.8625793
32: -33.2785568, 25.1601276, -33.3165512, 25.2510967, -58.5296478, 58.4766731
33: -54.7571335, 33.3657455, -54.8678284, 33.5287552, -88.2858887, 88.2335663
34: -47.4109688, 23.6846123, -47.4853821, 23.7860355, -71.1970062, 71.1699982
35: -43.2674866, 28.1140480, -43.3482246, 28.2426376, -71.5101166, 71.4622726
36: -41.4609680, 31.2176399, -41.4956970, 31.2891960, -72.7501678, 72.7133331
37: -60.9382439, 28.2316456, -61.0196228, 28.3002090, -89.2384491, 89.2512665
38: -53.1773224, 40.6603470, -53.2478867, 40.7454681, -93.9227905, 93.9082336
39: -62.2134895, 34.8178978, -62.2855606, 34.9526863, -97.1661682, 97.1034546
40: -54.8734779, 23.4387970, -54.9868698, 23.5046921, -78.3781662, 78.4256592
41: -35.5279694, 21.6656475, -35.5738564, 21.7221756, -57.2501373, 57.2395020
42: -29.8083000, 19.9452744, -29.8626366, 20.0422554, -49.8505554, 49.8079071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3155077, upper bound: 29.3258342
time: 69.82 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3155077, upper bound: 29.3377796
time: 73.09 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -46.4176712, 30.3987484, -46.4198647, 30.3258057, -76.7434769, 76.8186111
1: -27.7949390, 28.7760086, -27.7995491, 28.7404728, -56.5354118, 56.5755577
2: -24.6435394, 25.6989651, -24.6511898, 25.5898609, -50.2333984, 50.3501549
3: -30.7728481, 28.7910480, -30.7829914, 28.6711273, -59.4439697, 59.5740395
4: -29.2125015, 27.3574753, -29.2163811, 27.2801628, -56.4926529, 56.5738564
5: -30.3239956, 31.3072681, -30.3306561, 31.1297531, -61.4537430, 61.6379242
6: -36.4911423, 22.8314571, -36.4607925, 22.8179245, -59.3090591, 59.2922516
7: -35.6869583, 32.0693130, -35.6967506, 31.9961605, -67.6831207, 67.7660675
8: -38.5442467, 30.1314335, -38.5452194, 29.9994259, -68.5436630, 68.6766510
9: -31.9274235, 27.0334930, -31.9594097, 27.0313225, -58.9587402, 58.9929047
10: -44.0839958, 37.8972740, -44.1334991, 37.8813782, -81.9653778, 82.0307770
11: -42.7637100, 33.3562965, -42.7932701, 33.3473358, -76.1110382, 76.1495590
12: -38.3582077, 35.7987137, -38.1222000, 35.7968216, -74.1550293, 73.9209061
13: -40.1936760, 43.5459404, -40.1614075, 43.5618286, -83.7555084, 83.7073517
14: -66.3265152, 23.4939766, -66.3288116, 23.4829292, -89.8094482, 89.8227844
15: -36.8539238, 25.3046379, -36.8591194, 25.2598591, -62.1137848, 62.1637497
16: -45.9726944, 31.7072201, -45.9934196, 31.6950722, -77.6677551, 77.7006378
17: -66.2215652, 42.1488113, -66.1977234, 42.1556473, -108.3772125, 108.3465347
18: -38.0095406, 33.6326904, -38.0215836, 33.6229172, -71.6324539, 71.6542664
19: -31.9927273, 20.5583267, -31.9691944, 20.5598297, -52.5525589, 52.5275192
20: -27.4371109, 20.8891220, -27.4207382, 20.8795204, -48.3166313, 48.3098602
21: -38.5591507, 23.8258095, -38.5355530, 23.8127346, -62.3718796, 62.3613586
22: -37.5345383, 25.1509247, -37.5558586, 25.1418667, -62.6764069, 62.7067795
23: -31.5990486, 22.5368309, -31.6014252, 22.5254936, -54.1245422, 54.1382561
24: -32.5786819, 23.7236977, -32.6153488, 23.7167797, -56.2954636, 56.3390465
25: -29.9514637, 28.3395252, -29.9478168, 28.3345222, -58.2859879, 58.2873421
26: -47.0091934, 33.7444382, -46.9001389, 33.7346725, -80.7438507, 80.6445770
27: -38.5554199, 22.4562187, -38.6239128, 22.4810543, -61.0364685, 61.0801277
28: -31.7999249, 26.0294800, -31.8066044, 26.0557575, -57.8556747, 57.8360748
29: -39.0801506, 24.9813786, -39.1137962, 24.9704399, -64.0505829, 64.0951767
30: -37.6730270, 26.4621181, -37.7257309, 26.4489326, -64.1219635, 64.1878510
31: -36.2768326, 25.6543903, -36.2816124, 25.6477337, -61.9245682, 61.9360046
32: -33.4415855, 25.3050156, -33.3324394, 25.3037319, -58.7453156, 58.6374550
33: -54.9755707, 33.4734421, -54.9350891, 33.5437508, -88.5193176, 88.4085312
34: -47.5442238, 23.7755356, -47.5245094, 23.7970695, -71.3412933, 71.3000488
35: -43.4190521, 28.1864758, -43.3938179, 28.2523613, -71.6714096, 71.5802917
36: -41.5553513, 31.2818451, -41.5124397, 31.3097801, -72.8651199, 72.7942810
37: -61.0951347, 28.2955704, -61.0572510, 28.3148422, -89.4099731, 89.3528137
38: -53.3216400, 40.7698975, -53.2776909, 40.7795982, -94.1012421, 94.0475922
39: -62.3793144, 34.8738632, -62.3233261, 34.9607315, -97.3400421, 97.1971893
40: -55.0494957, 23.4916954, -55.0377808, 23.5121040, -78.5615997, 78.5294800
41: -35.6257401, 21.7478638, -35.5953445, 21.7467537, -57.3724823, 57.3432045
42: -29.9184418, 20.1059399, -29.8778191, 20.1002541, -50.0186920, 49.9837494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1625
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1686
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 518

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3377795, upper bound: 29.3258342
time: 80.66 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3377795, upper bound: 29.3377796
time: 73.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 156.67 seconds
IS_B2_A2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 156.67
Output dim: 2, lower bound: -29.2892864, upper bound: 29.3230069
IS_B2_A2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 156.67
Output dim: 2, lower bound: -29.2455131, upper bound: 29.2853131
IS_B2_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 156.67
Output dim: 2, lower bound: -29.2749319, upper bound: 29.3377528
IS_B2_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 156.67
Output dim: 2, lower bound: -29.2858441, upper bound: 29.3377528
IS_B2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 156.67
Output dim: 2, lower bound: -29.3377796, upper bound: 29.2872267
IS_B2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 156.67
Output dim: 2, lower bound: -29.3377796, upper bound: 29.2993193
IS_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 156.67
Output dim: 2, lower bound: -29.3155077, upper bound: 29.3258342
IS_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 156.67
Output dim: 2, lower bound: -29.3155077, upper bound: 29.3377796
IS_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 156.67
Output dim: 2, lower bound: -29.3377795, upper bound: 29.3258342
IS_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 156.67
Output dim: 2, lower bound: -29.3377795, upper bound: 29.3377796

## BFS IS instance: IS_B2_A2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -46.0529594, 30.1673698, -46.2928085, 30.2837086, -76.3366699, 76.4601746
1: -27.5730190, 28.6148415, -27.7251282, 28.7033348, -56.2763519, 56.3399620
2: -24.2821083, 25.4509487, -24.5110149, 25.5620842, -49.8441925, 49.9619637
3: -30.3929691, 28.5012455, -30.6352806, 28.6303387, -59.0233078, 59.1365242
4: -28.7690811, 27.1122208, -29.0442982, 27.2476673, -56.0167465, 56.1565170
5: -29.9475918, 30.9384727, -30.1867123, 31.0848732, -61.0324631, 61.1251793
6: -36.2646675, 22.5362110, -36.4095688, 22.7135830, -58.9782486, 58.9457779
7: -35.3738785, 31.8587437, -35.5955887, 31.9494400, -67.3233185, 67.4543228
8: -38.0989304, 29.8458862, -38.3746109, 29.9589443, -68.0578766, 68.2204971
9: -31.7405548, 26.7037354, -31.9008827, 26.9222469, -58.6627960, 58.6046143
10: -43.7270508, 37.2476082, -44.0669479, 37.6436539, -81.3707047, 81.3145523
11: -42.3684692, 32.7040863, -42.7340202, 33.0866508, -75.4551239, 75.4381027
12: -37.6939392, 34.9261398, -38.0665054, 35.4503708, -73.1443100, 72.9926453
13: -39.9960556, 43.1460114, -40.1216202, 43.4243279, -83.4203796, 83.2676315
14: -65.7913666, 22.8815594, -66.2406616, 23.2357979, -89.0271606, 89.1222229
15: -36.3779030, 25.0679226, -36.6846199, 25.2140160, -61.5919037, 61.7525406
16: -45.6938438, 31.2403755, -45.9233704, 31.5262070, -77.2200470, 77.1637421
17: -65.6695633, 41.2457542, -66.1286697, 41.7908554, -107.4604187, 107.3744202
18: -37.6892509, 33.3573685, -37.9469566, 33.5259933, -71.2152405, 71.3043213
19: -31.7479954, 20.3186684, -31.9198570, 20.4678612, -52.2158585, 52.2385254
20: -27.1821308, 20.6267776, -27.3742676, 20.7801800, -47.9623108, 48.0010452
21: -38.2172318, 23.4149723, -38.4842262, 23.6558342, -61.8730659, 61.8992004
22: -37.2530746, 24.7825089, -37.4842377, 25.0133247, -62.2663956, 62.2667465
23: -31.3984356, 22.3209839, -31.5570087, 22.4480572, -53.8464928, 53.8779907
24: -32.2936401, 23.5921745, -32.5204887, 23.6930771, -55.9867172, 56.1126633
25: -29.7278290, 28.1343842, -29.8837662, 28.2709732, -57.9987946, 58.0181503
26: -46.5387192, 33.2402802, -46.8376732, 33.5476608, -80.0863800, 80.0779419
27: -38.2309418, 22.3374882, -38.5189819, 22.4549999, -60.6859436, 60.8564644
28: -31.6043549, 25.8727036, -31.7612934, 25.9986229, -57.6029778, 57.6339951
29: -38.8472939, 24.5222893, -39.0653305, 24.7905273, -63.6378212, 63.5876198
30: -37.4743080, 26.1642742, -37.6766396, 26.3428974, -63.8171997, 63.8409119
31: -35.9975662, 25.4516983, -36.2184334, 25.5756531, -61.5732193, 61.6701279
32: -33.1329880, 24.9469051, -33.2842255, 25.1636333, -58.2966232, 58.2311325
33: -54.4386139, 33.2144051, -54.7422905, 33.4948044, -87.9334106, 87.9566956
34: -47.1499405, 23.5752277, -47.3826904, 23.7584724, -70.9084015, 70.9579163
35: -43.0424805, 28.0199642, -43.2591095, 28.2208843, -71.2633667, 71.2790756
36: -41.3374977, 31.1046162, -41.4554825, 31.2473736, -72.5848694, 72.5600967
37: -60.7207947, 28.1046486, -60.9445686, 28.2551250, -88.9759216, 89.0492096
38: -52.9838715, 40.5342865, -53.1816406, 40.7009201, -93.6847916, 93.7159271
39: -61.9784164, 34.7220154, -62.1983337, 34.9259491, -96.9043655, 96.9203491
40: -54.6044998, 23.3637943, -54.8859024, 23.4810715, -78.0855637, 78.2496948
41: -35.3880386, 21.5306282, -35.5269775, 21.6710358, -57.0590744, 57.0576057
42: -29.6860809, 19.6829472, -29.8283501, 19.9370670, -49.6231461, 49.5112953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=254, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=476, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1671
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1672

## Relational analysis of IS_B2_A2_A1_B2_A1_A1_B1

### Relational analysis result of IS_B2_A2_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2749319, upper bound: 29.3225611
time: 70.56 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2749319, upper bound: 29.3377528
time: 87.93 seconds

## BFS IS instance: IS_B2_A2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -46.1779175, 30.2446404, -46.3286743, 30.2978687, -76.4757843, 76.5733109
1: -27.6494446, 28.6716290, -27.7455139, 28.7140236, -56.3634644, 56.4171448
2: -24.4156322, 25.5478134, -24.5596352, 25.5701370, -49.9857635, 50.1074448
3: -30.5155945, 28.6021824, -30.6789532, 28.6438560, -59.1594505, 59.2811356
4: -28.9295864, 27.2234936, -29.1012497, 27.2587833, -56.1883621, 56.3247452
5: -30.0721550, 31.0922623, -30.2313843, 31.0995483, -61.1717033, 61.3236427
6: -36.3683243, 22.6417522, -36.4261093, 22.7443352, -59.1126595, 59.0678635
7: -35.4948196, 31.9200649, -35.6289825, 31.9607143, -67.4555359, 67.5490417
8: -38.2643356, 29.9693203, -38.4330330, 29.9716091, -68.2359467, 68.4023438
9: -31.8106270, 26.8530960, -31.9158306, 26.9681931, -58.7788200, 58.7689285
10: -43.8835487, 37.5224190, -44.0876389, 37.7374878, -81.6210251, 81.6100616
11: -42.5455666, 32.9151001, -42.7526093, 33.1639061, -75.7094574, 75.6677094
12: -38.0205994, 35.2591858, -38.0828629, 35.5770721, -73.5976715, 73.3420486
13: -40.1053391, 43.3155441, -40.1342545, 43.4770355, -83.5823746, 83.4497986
14: -66.0511627, 23.1060467, -66.2685394, 23.3231201, -89.3742828, 89.3745804
15: -36.5402107, 25.1893845, -36.7363625, 25.2295895, -61.7697906, 61.9257469
16: -45.8174744, 31.4109211, -45.9479752, 31.5799427, -77.3974152, 77.3588867
17: -65.9407654, 41.5578194, -66.1485291, 41.9104996, -107.8512650, 107.7063446
18: -37.8038902, 33.4470062, -37.9674072, 33.5517578, -71.3556519, 71.4144058
19: -31.8472290, 20.3773861, -31.9369621, 20.4856720, -52.3329010, 52.3143425
20: -27.2916241, 20.7077408, -27.3897858, 20.8062706, -48.0978928, 48.0975266
21: -38.3692284, 23.5366364, -38.5014839, 23.6964855, -62.0657043, 62.0381203
22: -37.3533096, 24.9076710, -37.5030289, 25.0497761, -62.4030838, 62.4106979
23: -31.4844093, 22.3805389, -31.5725803, 22.4637146, -53.9481239, 53.9531174
24: -32.4320374, 23.6732407, -32.5631371, 23.6999035, -56.1319427, 56.2363739
25: -29.8249741, 28.2032146, -29.9065170, 28.2845287, -58.1094971, 58.1097260
26: -46.7014809, 33.4026794, -46.8564186, 33.6032639, -80.3047485, 80.2590942
27: -38.3830032, 22.4000301, -38.5637817, 22.4616241, -60.8446198, 60.9638062
28: -31.6876030, 25.9057159, -31.7776108, 26.0065002, -57.6941032, 57.6833267
29: -38.9342575, 24.6679935, -39.0805130, 24.8431892, -63.7774429, 63.7485046
30: -37.5599747, 26.2523289, -37.6937943, 26.3665848, -63.9265594, 63.9461174
31: -36.1184158, 25.5010757, -36.2415771, 25.5874863, -61.7059021, 61.7426529
32: -33.2948952, 25.0907059, -33.3000412, 25.2160225, -58.5109024, 58.3907394
33: -54.6560020, 33.3216667, -54.8093185, 33.5098267, -88.1658173, 88.1309814
34: -47.2822113, 23.6663857, -47.4216919, 23.7694931, -71.0517044, 71.0880737
35: -43.1930313, 28.0920982, -43.3044853, 28.2305870, -71.4236145, 71.3965759
36: -41.4305229, 31.1682663, -41.4721375, 31.2673607, -72.6978836, 72.6404037
37: -60.8743706, 28.1659393, -60.9813461, 28.2692242, -89.1435928, 89.1472855
38: -53.1258888, 40.6373596, -53.2111816, 40.7323608, -93.8582458, 93.8485413
39: -62.1431274, 34.7780762, -62.2349281, 34.9339294, -97.0770416, 97.0130005
40: -54.7790909, 23.4164696, -54.9365273, 23.4884167, -78.2675095, 78.3529968
41: -35.4843216, 21.6106033, -35.5482559, 21.6954937, -57.1798096, 57.1588554
42: -29.7918949, 19.8409290, -29.8435116, 19.9947186, -49.7866135, 49.6844406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=254, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=476, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 1690
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1621

## Relational analysis of IS_B2_A2_A1_B2_A1_A2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2823104, upper bound: 29.3025391
time: 71.62 seconds

## Relational analysis of IS_B2_A2_A1_B2_A1_A2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2844040, upper bound: 29.3363289
time: 74.74 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -46.2369003, 30.2931747, -46.0449638, 30.1439381, -76.3808365, 76.3381348
1: -27.6799583, 28.6912041, -27.5583744, 28.5791054, -56.2590637, 56.2495728
2: -24.4655094, 25.5828056, -24.2590561, 25.4634209, -49.9289246, 49.8418617
3: -30.6069622, 28.6634369, -30.4053593, 28.4949684, -59.1019287, 59.0687943
4: -28.9899292, 27.2260475, -28.7205467, 27.1026974, -56.0926285, 55.9465866
5: -30.1589203, 31.1228733, -29.9567089, 30.9345894, -61.0935097, 61.0795822
6: -36.3558578, 22.6711712, -36.2353973, 22.4660835, -58.8219414, 58.9065666
7: -35.5426903, 31.9722919, -35.3802757, 31.8147011, -67.3573837, 67.3525696
8: -38.3090134, 29.9806938, -38.0324402, 29.7969666, -68.1059799, 68.0131378
9: -31.8223209, 26.8285675, -31.7173424, 26.6275806, -58.4499016, 58.5459099
10: -43.8890038, 37.4890594, -43.6880150, 37.0327911, -80.9217987, 81.1770782
11: -42.5480347, 32.9942627, -42.4261475, 32.5541496, -75.1021881, 75.4204102
12: -37.9987984, 35.3322792, -37.6802559, 34.7620888, -72.7608871, 73.0125351
13: -40.0656013, 43.2999268, -39.9571266, 43.0486717, -83.1142731, 83.2570419
14: -66.0264816, 23.1308022, -65.8335953, 22.6869717, -88.7134552, 88.9644012
15: -36.5856781, 25.1566505, -36.2785110, 25.0429001, -61.6285553, 61.4351616
16: -45.8110466, 31.4171619, -45.6660805, 31.0908165, -76.9018631, 77.0832367
17: -65.9190826, 41.6555672, -65.7842560, 41.0494347, -106.9685059, 107.4398117
18: -37.8565636, 33.5087242, -37.7088165, 33.3577499, -71.2143097, 71.2175446
19: -31.8700314, 20.4525299, -31.7653770, 20.3301640, -52.2001953, 52.2179031
20: -27.3032322, 20.7546101, -27.1978111, 20.5961876, -47.8994217, 47.9524193
21: -38.3800354, 23.6226997, -38.2604713, 23.3759003, -61.7559357, 61.8831711
22: -37.3860779, 24.9757309, -37.2518692, 24.7765484, -62.1626129, 62.2275963
23: -31.4898415, 22.4274940, -31.4060402, 22.2999573, -53.7897949, 53.8335342
24: -32.3881912, 23.6309052, -32.2227859, 23.5825729, -55.9707565, 55.8536911
25: -29.8189621, 28.2311707, -29.6900806, 28.1256428, -57.9446030, 57.9212494
26: -46.8029022, 33.5107269, -46.6040611, 33.2258224, -80.0287170, 80.1147919
27: -38.3324394, 22.3760777, -38.1713104, 22.3193550, -60.6517944, 60.5473862
28: -31.6800842, 25.9569054, -31.5969334, 25.8661308, -57.5462151, 57.5538368
29: -38.9605408, 24.7595520, -38.8919945, 24.4927940, -63.4533348, 63.6515427
30: -37.5559616, 26.3081779, -37.4974022, 26.1232052, -63.6791687, 63.8055763
31: -36.1305885, 25.5593300, -36.0174408, 25.4477596, -61.5783463, 61.5767670
32: -33.2497406, 25.1101627, -33.0827255, 24.8733482, -58.1230774, 58.1928864
33: -54.6514168, 33.3445625, -54.2870598, 33.2702637, -87.9216766, 87.6316223
34: -47.3099670, 23.6647568, -47.0447235, 23.5613384, -70.8712997, 70.7094803
35: -43.1749916, 28.0970154, -42.9063263, 28.0197544, -71.1947403, 71.0033417
36: -41.3994675, 31.2008438, -41.2401733, 31.1092834, -72.5087509, 72.4410172
37: -60.8551674, 28.2089520, -60.5932198, 28.1126976, -88.9678650, 88.8021698
38: -53.1092339, 40.6534615, -52.8973618, 40.5328293, -93.6420593, 93.5508270
39: -62.1234627, 34.8016434, -61.8466110, 34.8073997, -96.9308624, 96.6482544
40: -54.7727509, 23.4215851, -54.4709969, 23.3668633, -78.1396027, 77.8925781
41: -35.4790115, 21.6402798, -35.3245316, 21.5192261, -56.9982376, 56.9648094
42: -29.7775574, 19.8745136, -29.6531849, 19.5935383, -49.3710938, 49.5276985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=254, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2954423, upper bound: 29.2841093
time: 71.66 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3362408, upper bound: 29.2842296
time: 74.48 seconds

## BFS IS instance: IS_B2_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -46.2789993, 30.3090363, -46.1954002, 30.2305851, -76.5095825, 76.5044327
1: -27.7045898, 28.7049122, -27.6563625, 28.6466141, -56.3511963, 56.3612747
2: -24.5057011, 25.5936432, -24.3850327, 25.5489712, -50.0546722, 49.9786758
3: -30.6547527, 28.6772537, -30.5458641, 28.6288586, -59.2835999, 59.2231102
4: -29.0476818, 27.2369194, -28.8903294, 27.2433586, -56.2910309, 56.1272507
5: -30.2025032, 31.1402912, -30.0860176, 31.0956516, -61.2981567, 61.2263069
6: -36.3716316, 22.7172356, -36.3315353, 22.6142197, -58.9858513, 59.0487633
7: -35.5605240, 31.9851093, -35.4821396, 31.8808956, -67.4414215, 67.4672470
8: -38.3644524, 29.9954662, -38.2024651, 29.9188747, -68.2833176, 68.1979294
9: -31.8480797, 26.8525696, -31.8062553, 26.7442131, -58.5922928, 58.6588249
10: -43.9126396, 37.5541077, -43.8179665, 37.2578125, -81.1704559, 81.3720703
11: -42.5696220, 33.0987930, -42.6929665, 32.8345451, -75.4041672, 75.7917557
12: -38.0202484, 35.4430809, -38.0038643, 35.0611877, -73.0814285, 73.4469452
13: -40.0805092, 43.3513756, -40.0612030, 43.2230034, -83.3035126, 83.4125748
14: -66.0512314, 23.2221012, -66.1202087, 22.9291153, -88.9803467, 89.3423080
15: -36.6625328, 25.1704769, -36.5021400, 25.2099934, -61.8725128, 61.6726112
16: -45.8336525, 31.4822712, -45.8048248, 31.3066502, -77.1403046, 77.2870941
17: -65.9425125, 41.7990646, -66.1582794, 41.4224625, -107.3649750, 107.9573441
18: -37.8755760, 33.5345840, -37.8179283, 33.4596634, -71.3352356, 71.3525085
19: -31.8866596, 20.4968987, -31.9359703, 20.4569588, -52.3436127, 52.4328651
20: -27.3172112, 20.7977543, -27.3324680, 20.7226524, -48.0398636, 48.1302223
21: -38.3972816, 23.6894608, -38.4807854, 23.5649319, -61.9622116, 62.1702461
22: -37.4098129, 25.0253830, -37.3737030, 24.9373856, -62.3471985, 62.3990860
23: -31.5045319, 22.4631348, -31.5249939, 22.4086952, -53.9132233, 53.9881287
24: -32.4160690, 23.6399136, -32.3374405, 23.6394005, -56.0554657, 55.9773560
25: -29.8406353, 28.2623405, -29.8017693, 28.2294617, -58.0700989, 58.0641098
26: -46.8240433, 33.5606384, -46.7741241, 33.3900146, -80.2140579, 80.3347473
27: -38.3572311, 22.3868198, -38.2786255, 22.3697987, -60.7270203, 60.6654396
28: -31.6940384, 25.9867477, -31.7054482, 25.9548492, -57.6488876, 57.6921921
29: -38.9717560, 24.8266792, -39.0220947, 24.6787815, -63.6505280, 63.8487740
30: -37.5717316, 26.3453503, -37.5987244, 26.2501087, -63.8218384, 63.9440765
31: -36.1500778, 25.5949688, -36.1624680, 25.5540581, -61.7041321, 61.7574387
32: -33.2632904, 25.1578236, -33.2139587, 25.0155163, -58.2788086, 58.3717804
33: -54.7212067, 33.3580399, -54.5212555, 33.3871918, -88.1083984, 87.8792953
34: -47.3730240, 23.6770172, -47.2348785, 23.7320042, -71.1050262, 70.9118958
35: -43.2216110, 28.1053333, -43.0680313, 28.1048393, -71.3264465, 71.1733551
36: -41.4129562, 31.2242260, -41.3308487, 31.1936607, -72.6066132, 72.5550766
37: -60.8982773, 28.2237625, -60.7756729, 28.1964340, -89.0947113, 88.9994354
38: -53.1312180, 40.6711693, -53.0322418, 40.6244621, -93.7556763, 93.7034149
39: -62.1724586, 34.8046494, -62.0466347, 34.8678436, -97.0402985, 96.8512802
40: -54.8384819, 23.4329357, -54.6793442, 23.4834042, -78.3218842, 78.1122742
41: -35.5009995, 21.6683578, -35.4259644, 21.6195431, -57.1205292, 57.0943184
42: -29.7967606, 19.9278488, -29.7645531, 19.7510624, -49.5478210, 49.6924019

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=254, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1753
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1751
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1735
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1557
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1750
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: B, layer: 1, pos: 1708
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1679
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1647
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 1690

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A1

### Relational analysis result of IS_B2_A2_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.2954423, upper bound: 29.2961671
time: 68.91 seconds

## Relational analysis of IS_B2_A2_A2_B1_B1_B2_A2

### Relational analysis result of IS_B2_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3362408, upper bound: 29.2962305
time: 80.76 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -46.2278976, 30.2987061, -46.2541389, 30.2629471, -76.4908447, 76.5528412
1: -27.6803818, 28.6965332, -27.7035255, 28.6814384, -56.3618164, 56.4000587
2: -24.4564857, 25.5843143, -24.4873562, 25.5470810, -50.0035629, 50.0716705
3: -30.5856361, 28.6654415, -30.6037922, 28.6114426, -59.1970749, 59.2692337
4: -28.9755363, 27.2296810, -28.9929180, 27.2335587, -56.2090912, 56.2225914
5: -30.1383591, 31.1252193, -30.1579876, 31.0613480, -61.1996994, 61.2832069
6: -36.3611679, 22.6543388, -36.3889961, 22.6459637, -59.0071259, 59.0433350
7: -35.5296288, 31.9773769, -35.5844040, 31.9178619, -67.4474945, 67.5617828
8: -38.3073387, 29.9852581, -38.3309250, 29.9373856, -68.2447205, 68.3161774
9: -31.8215485, 26.8413677, -31.8676090, 26.8972301, -58.7187805, 58.7089767
10: -43.8937035, 37.5341187, -44.0395241, 37.5959282, -81.4896317, 81.5736389
11: -42.5497627, 33.0115128, -42.7067451, 32.9770203, -75.5267792, 75.7182541
12: -37.9984512, 35.3257217, -38.0359116, 35.3627892, -73.3612366, 73.3616333
13: -40.0621033, 43.3050766, -40.0944061, 43.3532295, -83.4153290, 83.3994827
14: -66.0221176, 23.1544189, -66.2109604, 23.1499596, -89.1720734, 89.3653793
15: -36.5815582, 25.1611595, -36.5805969, 25.1969719, -61.7785187, 61.7417564
16: -45.8121643, 31.4392509, -45.8915329, 31.4404106, -77.2525711, 77.3307800
17: -65.9124451, 41.6572533, -66.0976868, 41.6388626, -107.5513077, 107.7549362
18: -37.8523674, 33.5036201, -37.9095802, 33.5135231, -71.3658752, 71.4132004
19: -31.8701591, 20.4431572, -31.9006615, 20.4180279, -52.2881851, 52.3438187
20: -27.3048992, 20.7522774, -27.3583126, 20.7309322, -48.0358315, 48.1105881
21: -38.3819923, 23.6211357, -38.4635010, 23.5884895, -61.9704819, 62.0846367
22: -37.3894463, 24.9624329, -37.4415894, 24.9675293, -62.3569756, 62.4040222
23: -31.4917107, 22.4303265, -31.5393772, 22.4075317, -53.8992424, 53.9697037
24: -32.3961067, 23.6297264, -32.4800377, 23.6815758, -56.0776825, 56.1097641
25: -29.8181362, 28.2312393, -29.8496761, 28.2366638, -58.0547943, 58.0809174
26: -46.8031311, 33.4972534, -46.8065910, 33.5174255, -80.3205566, 80.3038406
27: -38.3548393, 22.3768635, -38.4783592, 22.4365044, -60.7913437, 60.8552208
28: -31.6964474, 25.9563427, -31.7457104, 25.9605522, -57.6569977, 57.7020531
29: -38.9672775, 24.7464256, -39.0425415, 24.7305717, -63.6978493, 63.7889671
30: -37.5620689, 26.3109760, -37.6539612, 26.2955894, -63.8576584, 63.9649353
31: -36.1279449, 25.5591068, -36.1973572, 25.5347977, -61.6627426, 61.7564583
32: -33.2550049, 25.0967236, -33.2645798, 25.1126862, -58.3676910, 58.3613052
33: -54.6660881, 33.3425179, -54.6663895, 33.4778938, -88.1439819, 88.0089111
34: -47.3317299, 23.6648483, -47.3091621, 23.7425900, -71.0743179, 70.9740143
35: -43.2037048, 28.0990105, -43.2069321, 28.2097855, -71.4134903, 71.3059387
36: -41.4297333, 31.1852913, -41.4266090, 31.2192516, -72.6489716, 72.6118927
37: -60.8764877, 28.2057114, -60.8822174, 28.2437515, -89.1202393, 89.0879211
38: -53.1295280, 40.6303482, -53.1445236, 40.6790009, -93.8085175, 93.7748718
39: -62.1436119, 34.7984467, -62.1313744, 34.9102211, -97.0538330, 96.9298096
40: -54.7904434, 23.4177399, -54.8047066, 23.4589176, -78.2493591, 78.2224426
41: -35.4920464, 21.6243420, -35.4944000, 21.6325779, -57.1246262, 57.1187363
42: -29.7824936, 19.8728943, -29.8052559, 19.8836117, -49.6661072, 49.6781502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=255, inp2_unstable=255, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=478, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1671
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1753
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1752
type: B, layer: 1, pos: 1752
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1751
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: B, layer: 1, pos: 1737
type: A, layer: 1, pos: 1624
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1624
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1722
type: B, layer: 1, pos: 1722
type: A, layer: 1, pos: 1707
type: B, layer: 1, pos: 1707
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1661
type: B, layer: 1, pos: 1661
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1681
type: B, layer: 1, pos: 1681
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1665
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1692
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1560
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1553
type: B, layer: 1, pos: 1553
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1670
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1606
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1748
type: A, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1557
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1608
type: A, layer: 1, pos: 1608
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1598
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1736
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1736
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1686
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1645
type: B, layer: 1, pos: 1645
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1750
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1585
type: A, layer: 1, pos: 1585
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1675
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1675
type: A, layer: 1, pos: 1684
type: B, layer: 1, pos: 1684
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1575
type: A, layer: 1, pos: 1575
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1569
type: B, layer: 1, pos: 1569
type: A, layer: 1, pos: 1627
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1749
type: A, layer: 1, pos: 1749
type: B, layer: 1, pos: 1537
type: A, layer: 1, pos: 1537
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1614
type: A, layer: 1, pos: 1614
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1611
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1611
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1626
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1617
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1626
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1601
type: A, layer: 1, pos: 1601
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1607
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 771
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1642
type: B, layer: 1, pos: 1642
type: A, layer: 1, pos: 1539
type: B, layer: 1, pos: 1539
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 772
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1633
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1633
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 517
type: A, layer: 1, pos: 517
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1578
type: A, layer: 1, pos: 1578
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 780
type: B, layer: 1, pos: 780
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1631
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 518

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_B2_A2_A2_B2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2713816, upper bound: 29.3241963
time: 79.95 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3140392, upper bound: 29.3242211
time: 66.63 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 148.86 seconds
IS_B2_A2_A1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 148.86
Output dim: 2, lower bound: -29.2749319, upper bound: 29.3225611
IS_B2_A2_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 148.86
Output dim: 2, lower bound: -29.2749319, upper bound: 29.3377528
IS_B2_A2_A1_B2_A1_A2_A1, status: Status.VERIFIED, split count: 7, time: 148.86
Output dim: 2, lower bound: -29.2823104, upper bound: 29.3025391
IS_B2_A2_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 148.86
Output dim: 2, lower bound: -29.2844040, upper bound: 29.3363289
IS_B2_A2_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 148.86
Output dim: 2, lower bound: -29.2954423, upper bound: 29.2841093
IS_B2_A2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 148.86
Output dim: 2, lower bound: -29.3362408, upper bound: 29.2842296
IS_B2_A2_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 148.86
Output dim: 2, lower bound: -29.2954423, upper bound: 29.2961671
IS_B2_A2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 148.86
Output dim: 2, lower bound: -29.3362408, upper bound: 29.2962305
IS_B2_A2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 148.86
Output dim: 2, lower bound: -29.2713816, upper bound: 29.3241963
IS_B2_A2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 148.86
Output dim: 2, lower bound: -29.3140392, upper bound: 29.3242211
IS_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 148.86
Output dim: 2, lower bound: -29.3155077, upper bound: 29.3377796
IS_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 148.86
Output dim: 2, lower bound: -29.3377795, upper bound: 29.3258342
IS_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 148.86
Output dim: 2, lower bound: -29.3377795, upper bound: 29.3377796

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 75.89 + 3654.76 = 3730.65 seconds
