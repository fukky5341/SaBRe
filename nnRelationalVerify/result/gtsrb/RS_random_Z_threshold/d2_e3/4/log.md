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
execution time: IAR + RelationalAnalysis = 2.76 + 72.98 = 75.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -29.3529070, upper bound: 29.3529070

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 512

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1653

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3516587, upper bound: 29.3450697
time: 74.88 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3450697, upper bound: 29.3516587
time: 80.83 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 155.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 155.73
Output dim: 2, lower bound: -29.3516587, upper bound: 29.3450697
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 155.73
Output dim: 2, lower bound: -29.3450697, upper bound: 29.3516587

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1591

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3508995, upper bound: 29.3318184
time: 70.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3384046, upper bound: 29.3443059
time: 73.37 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3252920, upper bound: 29.3506816
time: 62.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3441000, upper bound: 29.3318677
time: 63.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 127.91 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 127.91
Output dim: 2, lower bound: -29.3508995, upper bound: 29.3318184
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 127.91
Output dim: 2, lower bound: -29.3384046, upper bound: 29.3443059
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 127.91
Output dim: 2, lower bound: -29.3252920, upper bound: 29.3506816
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 127.91
Output dim: 2, lower bound: -29.3441000, upper bound: 29.3318677

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1567

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3501799, upper bound: 29.3313840
time: 73.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3504592, upper bound: 29.3310758
time: 85.57 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1667

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1742

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3188416, upper bound: 29.3437197
time: 76.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3378356, upper bound: 29.3249079
time: 72.99 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1697

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1602

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3243382, upper bound: 29.3502612
time: 69.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3243382, upper bound: 29.3497442
time: 74.28 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 521

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1624

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3358790, upper bound: 29.3306398
time: 66.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3429164, upper bound: 29.3234595
time: 77.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 145.71 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 145.71
Output dim: 2, lower bound: -29.3501799, upper bound: 29.3313840
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 145.71
Output dim: 2, lower bound: -29.3504592, upper bound: 29.3310758
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 145.71
Output dim: 2, lower bound: -29.3188416, upper bound: 29.3437197
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 145.71
Output dim: 2, lower bound: -29.3378356, upper bound: 29.3249079
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 145.71
Output dim: 2, lower bound: -29.3243382, upper bound: 29.3502612
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 145.71
Output dim: 2, lower bound: -29.3243382, upper bound: 29.3497442
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 145.71
Output dim: 2, lower bound: -29.3358790, upper bound: 29.3306398
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 145.71
Output dim: 2, lower bound: -29.3429164, upper bound: 29.3234595

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 525

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1725

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3440103, upper bound: 29.3307249
time: 67.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3494497, upper bound: 29.3252426
time: 85.36 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1604

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1743

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3389539, upper bound: 29.3306177
time: 72.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3500067, upper bound: 29.3194947
time: 69.47 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1674

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1550

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3164860, upper bound: 29.3411876
time: 65.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3162900, upper bound: 29.3413616
time: 56.75 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1683

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1737

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3363993, upper bound: 29.3057731
time: 65.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -29.3186210, upper bound: 29.3234330
time: 78.56 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1759

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3196099, upper bound: 29.3456366
time: 66.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3196099, upper bound: 29.3456366
time: 57.00 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1583

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1739

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3234647, upper bound: 29.3464208
time: 87.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3215550, upper bound: 29.3488799
time: 73.52 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1614

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 781

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3354370, upper bound: 29.3306377
time: 63.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3358769, upper bound: 29.3301679
time: 76.61 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 781

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1598

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3329615, upper bound: 29.3230203
time: 63.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3424670, upper bound: 29.3135759
time: 62.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 128.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3440103, upper bound: 29.3307249
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3494497, upper bound: 29.3252426
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3389539, upper bound: 29.3306177
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3500067, upper bound: 29.3194947
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3164860, upper bound: 29.3411876
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3162900, upper bound: 29.3413616
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3363993, upper bound: 29.3057731
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3186210, upper bound: 29.3234330
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3196099, upper bound: 29.3456366
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3196099, upper bound: 29.3456366
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3234647, upper bound: 29.3464208
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3215550, upper bound: 29.3488799
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3354370, upper bound: 29.3306377
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3358769, upper bound: 29.3301679
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3329615, upper bound: 29.3230203
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 128.16
Output dim: 2, lower bound: -29.3424670, upper bound: 29.3135759

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1775

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1716

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3396566, upper bound: 29.3296864
time: 76.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3396566, upper bound: 29.3264037
time: 77.15 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1713

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1789

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3357779, upper bound: 29.3245301
time: 66.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3487296, upper bound: 29.3115790
time: 73.88 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1756

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 519

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3365749, upper bound: 29.3266035
time: 74.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3349034, upper bound: 29.3283178
time: 76.74 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1559

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1599

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3412038, upper bound: 29.3187184
time: 71.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3492664, upper bound: 29.3105558
time: 66.57 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1540

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1607

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3154048, upper bound: 29.3383529
time: 74.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3136908, upper bound: 29.3403324
time: 103.56 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1709

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1694

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3154024, upper bound: 29.3355488
time: 81.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3106766, upper bound: 29.3405232
time: 67.11 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1739

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 512

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3338687, upper bound: 29.3055144
time: 85.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3361402, upper bound: 29.3032408
time: 82.63 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1640

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1739

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3187282, upper bound: 29.3423393
time: 65.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3163109, upper bound: 29.3447687
time: 85.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=258, inp2_unstable=258, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=477, inp2_unstable=477, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 512
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1654

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1682

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3180845, upper bound: 29.3369199
time: 88.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -29.3108623, upper bound: 29.3440324
time: 83.11 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 173.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3396566, upper bound: 29.3296864
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3396566, upper bound: 29.3264037
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3357779, upper bound: 29.3245301
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3487296, upper bound: 29.3115790
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3365749, upper bound: 29.3266035
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3349034, upper bound: 29.3283178
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3412038, upper bound: 29.3187184
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3492664, upper bound: 29.3105558
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3154048, upper bound: 29.3383529
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3136908, upper bound: 29.3403324
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3154024, upper bound: 29.3355488
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3106766, upper bound: 29.3405232
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3338687, upper bound: 29.3055144
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3361402, upper bound: 29.3032408
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3187282, upper bound: 29.3423393
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3163109, upper bound: 29.3447687
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3180845, upper bound: 29.3369199
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 173.47
Output dim: 2, lower bound: -29.3108623, upper bound: 29.3440324
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 173.47
Output dim: 2, lower bound: -29.3234647, upper bound: 29.3464208
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 173.47
Output dim: 2, lower bound: -29.3215550, upper bound: 29.3488799
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 173.47
Output dim: 2, lower bound: -29.3354370, upper bound: 29.3306377
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 173.47
Output dim: 2, lower bound: -29.3358769, upper bound: 29.3301679
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 173.47
Output dim: 2, lower bound: -29.3329615, upper bound: 29.3230203
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 173.47
Output dim: 2, lower bound: -29.3424670, upper bound: 29.3135759

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 75.74 + 3582.40 = 3658.14 seconds
