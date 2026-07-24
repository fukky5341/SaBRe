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
execution time: IAR + RelationalAnalysis = 2.32 + 75.19 = 77.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -29.3529070, upper bound: 29.3529070

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3115163, upper bound: 29.3508470
time: 69.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3508470, upper bound: 29.3115163
time: 77.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 147.16 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 147.16
Output dim: 2, lower bound: -29.3115163, upper bound: 29.3508470
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 147.16
Output dim: 2, lower bound: -29.3508470, upper bound: 29.3115163

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2874170, upper bound: 29.3500190
time: 62.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3087096, upper bound: 29.3161680
time: 79.58 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3161680, upper bound: 29.3087096
time: 75.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3500190, upper bound: 29.2874170
time: 78.23 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 156.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 156.15
Output dim: 2, lower bound: -29.2874170, upper bound: 29.3500190
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 156.15
Output dim: 2, lower bound: -29.3087096, upper bound: 29.3161680
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 156.15
Output dim: 2, lower bound: -29.3161680, upper bound: 29.3087096
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 156.15
Output dim: 2, lower bound: -29.3500190, upper bound: 29.2874170

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2855702, upper bound: 29.3178784
time: 75.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2546554, upper bound: 29.3479894
time: 75.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3068423, upper bound: 29.2835499
time: 83.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2761102, upper bound: 29.3142370
time: 85.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3142370, upper bound: 29.2761102
time: 67.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2835499, upper bound: 29.3068423
time: 68.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1721

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3479894, upper bound: 29.2546554
time: 72.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3178784, upper bound: 29.2855702
time: 67.49 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 142.38 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 142.38
Output dim: 2, lower bound: -29.2855702, upper bound: 29.3178784
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 142.38
Output dim: 2, lower bound: -29.2546554, upper bound: 29.3479894
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 142.38
Output dim: 2, lower bound: -29.3068423, upper bound: 29.2835499
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 142.38
Output dim: 2, lower bound: -29.2761102, upper bound: 29.3142370
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 142.38
Output dim: 2, lower bound: -29.3142370, upper bound: 29.2761102
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 142.38
Output dim: 2, lower bound: -29.2835499, upper bound: 29.3068423
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 142.38
Output dim: 2, lower bound: -29.3479894, upper bound: 29.2546554
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 142.38
Output dim: 2, lower bound: -29.3178784, upper bound: 29.2855702

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2599199, upper bound: 29.3166681
time: 71.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2843412, upper bound: 29.2923191
time: 75.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2289473, upper bound: 29.3468208
time: 82.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2534181, upper bound: 29.3225390
time: 93.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2812909, upper bound: 29.2823347
time: 74.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3056382, upper bound: 29.2579580
time: 66.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2504746, upper bound: 29.3130447
time: 65.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2748833, upper bound: 29.2887334
time: 78.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2887334, upper bound: 29.2748833
time: 68.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3130447, upper bound: 29.2504746
time: 75.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2289473, upper bound: 29.3056382
time: 75.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2823347, upper bound: 29.2812909
time: 86.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3225390, upper bound: 29.2534181
time: 78.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3468208, upper bound: 29.2289473
time: 78.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2923191, upper bound: 29.2843412
time: 81.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3166681, upper bound: 29.2599199
time: 87.15 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 171.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.2599199, upper bound: 29.3166681
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.2843412, upper bound: 29.2923191
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.2289473, upper bound: 29.3468208
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.2534181, upper bound: 29.3225390
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.2812909, upper bound: 29.2823347
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.3056382, upper bound: 29.2579580
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.2504746, upper bound: 29.3130447
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.2748833, upper bound: 29.2887334
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.2887334, upper bound: 29.2748833
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.3130447, upper bound: 29.2504746
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.2289473, upper bound: 29.3056382
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.2823347, upper bound: 29.2812909
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.3225390, upper bound: 29.2534181
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.3468208, upper bound: 29.2289473
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.2923191, upper bound: 29.2843412
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 171.12
Output dim: 2, lower bound: -29.3166681, upper bound: 29.2599199

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2588926, upper bound: 29.2852995
time: 78.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2286143, upper bound: 29.3155501
time: 71.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2833161, upper bound: 29.2608797
time: 72.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2530742, upper bound: 29.2911980
time: 72.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2278837, upper bound: 29.3158613
time: 74.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.1972240, upper bound: 29.3457860
time: 65.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2523655, upper bound: 29.2915248
time: 76.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2217244, upper bound: 29.3215060
time: 76.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2802534, upper bound: 29.2507321
time: 82.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2499188, upper bound: 29.2812338
time: 73.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3046013, upper bound: 29.2262841
time: 67.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2743505, upper bound: 29.2568382
time: 68.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2493544, upper bound: 29.2817917
time: 66.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2186991, upper bound: 29.3119963
time: 64.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2737603, upper bound: 29.2573745
time: 65.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2431673, upper bound: 29.2876803
time: 70.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -46.4110565, 30.2163239, -46.4110565, 30.2163239, -76.6273804, 76.6273804
1: -27.7927971, 28.7016869, -27.7927971, 28.7016869, -56.4944839, 56.4944839
2: -24.6698627, 25.4683743, -24.6698627, 25.4683743, -50.1382370, 50.1382370
3: -30.8162289, 28.5304737, -30.8162289, 28.5304737, -59.3467026, 59.3467026
4: -29.2495136, 27.1315174, -29.2495136, 27.1315174, -56.3810310, 56.3810310
5: -30.3568039, 30.9656982, -30.3568039, 30.9656982, -61.3225021, 61.3225021
6: -36.3908463, 22.8847961, -36.3908463, 22.8847961, -59.2756424, 59.2756424
7: -35.6764565, 31.9750862, -35.6764565, 31.9750862, -67.6515350, 67.6515427
8: -38.5683517, 29.8972931, -38.5683517, 29.8972931, -68.4656296, 68.4656448
9: -31.9024200, 26.9918976, -31.9024200, 26.9918976, -58.8943100, 58.8943176
10: -43.9016304, 37.8932457, -43.9016304, 37.8932457, -81.7948761, 81.7948761
11: -42.4117699, 33.4206276, -42.4117699, 33.4206276, -75.8323975, 75.8323975
12: -37.6801910, 35.8725357, -37.6801910, 35.8725357, -73.5527267, 73.5527267
13: -40.1657639, 43.5063667, -40.1657639, 43.5063667, -83.6721344, 83.6721344
14: -65.9868469, 23.5374298, -65.9868469, 23.5374298, -89.5242615, 89.5242691
15: -36.8870697, 25.1650658, -36.8870697, 25.1650658, -62.0521355, 62.0521355
16: -45.9001236, 31.7527466, -45.9001236, 31.7527466, -77.6528702, 77.6528702
17: -65.7015381, 42.2335625, -65.7015381, 42.2335625, -107.9350891, 107.9351044
18: -37.8993378, 33.6553726, -37.8993378, 33.6553726, -71.5547028, 71.5547104
19: -31.8146629, 20.5727043, -31.8146629, 20.5727043, -52.3873596, 52.3873634
20: -27.2639866, 20.8974056, -27.2639866, 20.8974056, -48.1613922, 48.1613922
21: -38.2648811, 23.8429794, -38.2648811, 23.8429794, -62.1078568, 62.1078606
22: -37.4528580, 25.1443119, -37.4528580, 25.1443119, -62.5971603, 62.5971680
23: -31.4789429, 22.5381432, -31.4789429, 22.5381432, -54.0170860, 54.0170860
24: -32.5523491, 23.6669846, -32.5523491, 23.6669846, -56.2193298, 56.2193260
25: -29.8995991, 28.3100185, -29.8995991, 28.3100185, -58.2096176, 58.2096138
26: -46.5979881, 33.7923813, -46.5979881, 33.7923813, -80.3903656, 80.3903656
27: -38.5072021, 22.4410686, -38.5072021, 22.4410686, -60.9482727, 60.9482689
28: -31.6841717, 26.0362110, -31.6841717, 26.0362110, -57.7203827, 57.7203827
29: -38.9248161, 25.0170403, -38.9248161, 25.0170403, -63.9418564, 63.9418564
30: -37.5897484, 26.4760761, -37.5897484, 26.4760761, -64.0658188, 64.0658264
31: -36.1398888, 25.6558552, -36.1398888, 25.6558552, -61.7957306, 61.7957420
32: -33.2356720, 25.3364716, -33.2356720, 25.3364716, -58.5721436, 58.5721436
33: -54.9592018, 33.3062782, -54.9592018, 33.3062782, -88.2654724, 88.2654724
34: -47.5322418, 23.6270466, -47.5322418, 23.6270466, -71.1592865, 71.1592865
35: -43.4169197, 28.0877247, -43.4169197, 28.0877247, -71.5046387, 71.5046463
36: -41.4978714, 31.2779446, -41.4978714, 31.2779446, -72.7758179, 72.7758179
37: -61.0181274, 28.2798958, -61.0181274, 28.2798958, -89.2980194, 89.2980194
38: -53.2566757, 40.7401047, -53.2566757, 40.7401047, -93.9967804, 93.9967804
39: -62.3372192, 34.8142319, -62.3372192, 34.8142319, -97.1514511, 97.1514511
40: -55.0095062, 23.4481621, -55.0095062, 23.4481621, -78.4576645, 78.4576721
41: -35.5631027, 21.7710762, -35.5631027, 21.7710762, -57.3341675, 57.3341751
42: -29.7974949, 20.1372986, -29.7974949, 20.1372986, -49.9347839, 49.9347839

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2876803, upper bound: 29.2431674
time: 83.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.2573745, upper bound: 29.2737603
time: 77.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 163.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2588926, upper bound: 29.2852995
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2286143, upper bound: 29.3155501
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2833161, upper bound: 29.2608797
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2530742, upper bound: 29.2911980
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2278837, upper bound: 29.3158613
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.1972240, upper bound: 29.3457860
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2523655, upper bound: 29.2915248
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2217244, upper bound: 29.3215060
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2802534, upper bound: 29.2507321
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2499188, upper bound: 29.2812338
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.3046013, upper bound: 29.2262841
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2743505, upper bound: 29.2568382
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2493544, upper bound: 29.2817917
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2186991, upper bound: 29.3119963
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2737603, upper bound: 29.2573745
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2431673, upper bound: 29.2876803
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2876803, upper bound: 29.2431674
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 163.09
Output dim: 2, lower bound: -29.2573745, upper bound: 29.2737603
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 163.09
Output dim: 2, lower bound: -29.3130447, upper bound: 29.2504746
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 163.09
Output dim: 2, lower bound: -29.2289473, upper bound: 29.3056382
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 163.09
Output dim: 2, lower bound: -29.2823347, upper bound: 29.2812909
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 163.09
Output dim: 2, lower bound: -29.3225390, upper bound: 29.2534181
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 163.09
Output dim: 2, lower bound: -29.3468208, upper bound: 29.2289473
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 163.09
Output dim: 2, lower bound: -29.2923191, upper bound: 29.2843412
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 163.09
Output dim: 2, lower bound: -29.3166681, upper bound: 29.2599199

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 77.50 + 3634.94 = 3712.45 seconds
