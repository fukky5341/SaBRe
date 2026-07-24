## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 7200 seconds
Split limit: 100


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433)
1: (-44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771)
2: (-41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853)
3: (-50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607)
4: (-52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401)
5: (-51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330)
6: (-57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770)
7: (-56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538)
8: (-57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571)
9: (-51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400)
10: (-67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825)
11: (-65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368)
12: (-60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301)
13: (-64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040)
14: (-99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981)
15: (-60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367)
16: (-69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389)
17: (-100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553)
18: (-58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543)
19: (-47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812)
20: (-43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006)
21: (-59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802)
22: (-59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535)
23: (-48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775)
24: (-53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096)
25: (-51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741)
26: (-69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453)
27: (-58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407)
28: (-49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753)
29: (-60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447)
30: (-61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987)
31: (-58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673)
32: (-52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904)
33: (-77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047)
34: (-63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257)
35: (-60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658)
36: (-58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090)
37: (-88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163)
38: (-71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014)
39: (-81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295)
40: (-73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867)
41: (-54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439)
42: (-46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.71 + 84.31 = 87.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -78.5698195, upper bound: 78.5698195

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.5608937, upper bound: 78.4212832
time: 74.25 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.4212832, upper bound: 78.5608937
time: 65.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 140.24 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 140.24
Output dim: 12, lower bound: -78.5608937, upper bound: 78.4212832
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 140.24
Output dim: 12, lower bound: -78.4212832, upper bound: 78.5608937

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.5540347, upper bound: 78.3134036
time: 84.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.4010451, upper bound: 78.4080261
time: 74.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.4080261, upper bound: 78.4010451
time: 81.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3134036, upper bound: 78.5540347
time: 137.17 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 221.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 221.10
Output dim: 12, lower bound: -78.5540347, upper bound: 78.3134036
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 221.10
Output dim: 12, lower bound: -78.4010451, upper bound: 78.4080261
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 221.10
Output dim: 12, lower bound: -78.4080261, upper bound: 78.4010451
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 221.10
Output dim: 12, lower bound: -78.3134036, upper bound: 78.5540347

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.5440448, upper bound: 78.1716135
time: 93.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.4145101, upper bound: 78.3052576
time: 70.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3917215, upper bound: 78.2671466
time: 63.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2604089, upper bound: 78.3990237
time: 81.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3990237, upper bound: 78.2604089
time: 71.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2671466, upper bound: 78.3917216
time: 71.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3052576, upper bound: 78.4145101
time: 60.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1716135, upper bound: 78.5440448
time: 91.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 155.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 155.16
Output dim: 12, lower bound: -78.5440448, upper bound: 78.1716135
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 155.16
Output dim: 12, lower bound: -78.4145101, upper bound: 78.3052576
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 155.16
Output dim: 12, lower bound: -78.3917215, upper bound: 78.2671466
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 155.16
Output dim: 12, lower bound: -78.2604089, upper bound: 78.3990237
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 155.16
Output dim: 12, lower bound: -78.3990237, upper bound: 78.2604089
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 155.16
Output dim: 12, lower bound: -78.2671466, upper bound: 78.3917216
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 155.16
Output dim: 12, lower bound: -78.3052576, upper bound: 78.4145101
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 155.16
Output dim: 12, lower bound: -78.1716135, upper bound: 78.5440448

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.5360572, upper bound: 78.1098929
time: 68.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.4567173, upper bound: 78.1602329
time: 90.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.4067925, upper bound: 78.2411102
time: 65.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3267911, upper bound: 78.2936962
time: 81.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3773201, upper bound: 78.1785894
time: 75.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3397537, upper bound: 78.2600918
time: 76.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2462044, upper bound: 78.3109281
time: 98.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2083107, upper bound: 78.3916516
time: 81.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3916515, upper bound: 78.2083107
time: 66.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3109281, upper bound: 78.2462045
time: 155.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2600918, upper bound: 78.3397538
time: 67.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1785894, upper bound: 78.3773201
time: 66.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2936962, upper bound: 78.3267911
time: 61.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2411102, upper bound: 78.4067925
time: 74.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1602329, upper bound: 78.4567173
time: 73.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1098929, upper bound: 78.5360572
time: 75.81 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 151.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.5360572, upper bound: 78.1098929
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.4567173, upper bound: 78.1602329
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.4067925, upper bound: 78.2411102
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.3267911, upper bound: 78.2936962
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.3773201, upper bound: 78.1785894
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.3397537, upper bound: 78.2600918
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.2462044, upper bound: 78.3109281
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.2083107, upper bound: 78.3916516
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.3916515, upper bound: 78.2083107
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.3109281, upper bound: 78.2462045
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.2600918, upper bound: 78.3397538
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.1785894, upper bound: 78.3773201
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.2936962, upper bound: 78.3267911
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.2411102, upper bound: 78.4067925
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.1602329, upper bound: 78.4567173
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 151.95
Output dim: 12, lower bound: -78.1098929, upper bound: 78.5360572

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.5274749, upper bound: 78.0247173
time: 70.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3989485, upper bound: 78.0982038
time: 76.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.4480436, upper bound: 78.0753920
time: 69.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3187590, upper bound: 78.1474877
time: 69.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3934752, upper bound: 78.1026015
time: 76.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3224882, upper bound: 78.2342723
time: 85.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3135569, upper bound: 78.1577288
time: 79.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2420827, upper bound: 78.2870256
time: 149.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3706006, upper bound: 78.0937407
time: 76.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2498311, upper bound: 78.1658668
time: 74.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2091759, upper bound: 78.1756710
time: 73.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2091759, upper bound: 78.2471237
time: 85.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1621829, upper bound: 78.1723673
time: 68.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1621829, upper bound: 78.3024152
time: 69.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1970109, upper bound: 78.2538002
time: 71.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1223762, upper bound: 78.3828749
time: 77.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1723673, upper bound: 78.1223763
time: 76.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2538002, upper bound: 78.1970109
time: 65.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3024152, upper bound: 78.1621829
time: 76.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1723673, upper bound: 78.2331981
time: 67.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2471237, upper bound: 78.2091759
time: 77.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1756710, upper bound: 78.3344671
time: 73.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.0937407, upper bound: 78.2498311
time: 84.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.0937407, upper bound: 78.3706006
time: 72.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2870256, upper bound: 78.2420827
time: 73.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1577287, upper bound: 78.3135569
time: 69.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2342722, upper bound: 78.3224882
time: 87.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1026015, upper bound: 78.3934753
time: 65.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1474877, upper bound: 78.3187590
time: 71.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.0753920, upper bound: 78.4480436
time: 72.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1672

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.0982038, upper bound: 78.3989485
time: 66.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.0247172, upper bound: 78.5274749
time: 73.02 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 141.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.5274749, upper bound: 78.0247173
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.3989485, upper bound: 78.0982038
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.4480436, upper bound: 78.0753920
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.3187590, upper bound: 78.1474877
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.3934752, upper bound: 78.1026015
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.3224882, upper bound: 78.2342723
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.3135569, upper bound: 78.1577288
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.2420827, upper bound: 78.2870256
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.3706006, upper bound: 78.0937407
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.2498311, upper bound: 78.1658668
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.2091759, upper bound: 78.1756710
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.2091759, upper bound: 78.2471237
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.1621829, upper bound: 78.1723673
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.1621829, upper bound: 78.3024152
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.1970109, upper bound: 78.2538002
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.1223762, upper bound: 78.3828749
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.1723673, upper bound: 78.1223763
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.2538002, upper bound: 78.1970109
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.3024152, upper bound: 78.1621829
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.1723673, upper bound: 78.2331981
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.2471237, upper bound: 78.2091759
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.1756710, upper bound: 78.3344671
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.0937407, upper bound: 78.2498311
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.0937407, upper bound: 78.3706006
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.2870256, upper bound: 78.2420827
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.1577287, upper bound: 78.3135569
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.2342722, upper bound: 78.3224882
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.1026015, upper bound: 78.3934753
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.1474877, upper bound: 78.3187590
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.0753920, upper bound: 78.4480436
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.0982038, upper bound: 78.3989485
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 141.68
Output dim: 12, lower bound: -78.0247172, upper bound: 78.5274749

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.5179554, upper bound: 77.9409025
time: 89.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.4485039, upper bound: 78.0146321
time: 74.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3894615, upper bound: 78.0167660
time: 63.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3191446, upper bound: 78.0880433
time: 98.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3089036, upper bound: 77.9938074
time: 78.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3682735, upper bound: 78.0652153
time: 72.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3089036, upper bound: 78.0663626
time: 63.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1278798, upper bound: 78.1373398
time: 60.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3840744, upper bound: 78.0209948
time: 85.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3138851, upper bound: 78.0923653
time: 84.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3128912, upper bound: 78.1533769
time: 66.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2423646, upper bound: 78.2240568
time: 77.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3038080, upper bound: 78.0765471
time: 85.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2332223, upper bound: 78.1474214
time: 68.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2320285, upper bound: 78.2066595
time: 75.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1612550, upper bound: 78.2770975
time: 67.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3608833, upper bound: 78.0119756
time: 68.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2906471, upper bound: 78.0836405
time: 83.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2396780, upper bound: 78.0846784
time: 69.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1690785, upper bound: 78.1557820
time: 66.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3244604, upper bound: 78.0947775
time: 80.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2540248, upper bound: 78.1656931
time: 69.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1988230, upper bound: 78.1667624
time: 67.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1278798, upper bound: 78.2373979
time: 76.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2232237, upper bound: 78.0910823
time: 78.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.0810694, upper bound: 78.1621950
time: 71.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1519826, upper bound: 78.2219899
time: 78.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.0810694, upper bound: 78.2925038
time: 75.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -75.1733780, 52.9547653, -75.1733780, 52.9547653, -128.1281433, 128.1281433
1: -44.3970757, 40.4098015, -44.3970757, 40.4098015, -84.8068771, 84.8068771
2: -41.5878067, 41.3514786, -41.5878067, 41.3514786, -82.9392853, 82.9392853
3: -50.2898598, 45.9705048, -50.2898598, 45.9705048, -96.2603607, 96.2603607
4: -52.2501602, 48.7337799, -52.2501602, 48.7337799, -100.9839401, 100.9839401
5: -51.3968849, 49.9778481, -51.3968849, 49.9778481, -101.3747253, 101.3747330
6: -57.6121254, 41.8589630, -57.6121254, 41.8589630, -99.4710846, 99.4710770
7: -56.6549606, 45.5595894, -56.6549606, 45.5595894, -102.2145538, 102.2145538
8: -57.5071373, 53.9682312, -57.5071373, 53.9682312, -111.4753571, 111.4753571
9: -51.3820343, 47.4424095, -51.3820343, 47.4424095, -98.8244476, 98.8244400
10: -67.5261002, 60.0571785, -67.5261002, 60.0571785, -127.5832825, 127.5832825
11: -65.4784851, 46.4545517, -65.4784851, 46.4545517, -111.9330368, 111.9330368
12: -60.8207893, 61.3994446, -60.8207893, 61.3994446, -122.2202301, 122.2202301
13: -64.1892853, 57.3298225, -64.1892853, 57.3298225, -121.5191040, 121.5191040
14: -99.1619339, 48.8261642, -99.1619339, 48.8261642, -147.9880981, 147.9880981
15: -60.6473656, 43.4983673, -60.6473656, 43.4983673, -104.1457367, 104.1457367
16: -69.0556641, 47.0748787, -69.0556641, 47.0748787, -116.1305389, 116.1305389
17: -100.3084488, 54.1382103, -100.3084488, 54.1382103, -154.4466248, 154.4466553
18: -58.3502350, 49.3801270, -58.3502350, 49.3801270, -107.7303543, 107.7303543
19: -47.9990768, 35.0060043, -47.9990768, 35.0060043, -83.0050812, 83.0050812
20: -43.1188698, 36.8092461, -43.1188698, 36.8092461, -79.9281006, 79.9281006
21: -59.3408203, 40.4563599, -59.3408203, 40.4563599, -99.7971802, 99.7971802
22: -59.3365021, 40.5105515, -59.3365021, 40.5105515, -99.8470535, 99.8470535
23: -48.9799881, 40.1915970, -48.9799881, 40.1915970, -89.1715698, 89.1715775
24: -53.8048744, 39.5652351, -53.8048744, 39.5652351, -93.3701019, 93.3701096
25: -51.1027641, 45.5151138, -51.1027641, 45.5151138, -96.6178741, 96.6178741
26: -69.8466492, 61.4585114, -69.8466492, 61.4585114, -131.3051605, 131.3051453
27: -58.8220024, 40.7747383, -58.8220024, 40.7747383, -99.5967407, 99.5967407
28: -49.8538055, 45.0549698, -49.8538055, 45.0549698, -94.9087753, 94.9087753
29: -60.6387558, 35.2923889, -60.6387558, 35.2923889, -95.9311447, 95.9311447
30: -61.6308632, 44.9840431, -61.6308632, 44.9840431, -106.6149063, 106.6148987
31: -58.8089180, 39.6213531, -58.8089180, 39.6213531, -98.4302673, 98.4302673
32: -52.0444832, 39.6619034, -52.0444832, 39.6619034, -91.7063904, 91.7063904
33: -77.7976379, 50.2409630, -77.7976379, 50.2409630, -128.0386047, 128.0386047
34: -63.6413116, 42.4563103, -63.6413116, 42.4563103, -106.0976257, 106.0976257
35: -60.0114822, 42.2715836, -60.0114822, 42.2715836, -102.2830658, 102.2830658
36: -58.7284584, 43.0120544, -58.7284584, 43.0120544, -101.7405090, 101.7405090
37: -88.5325699, 50.3628387, -88.5325699, 50.3628387, -138.8954163, 138.8954163
38: -71.2892532, 52.7749481, -71.2892532, 52.7749481, -124.0642014, 124.0642014
39: -81.6367874, 44.3372421, -81.6367874, 44.3372421, -125.9740295, 125.9740295
40: -73.5369415, 39.7081490, -73.5369415, 39.7081490, -113.2450867, 113.2450867
41: -54.9820862, 44.1282578, -54.9820862, 44.1282578, -99.1103439, 99.1103439
42: -46.3856926, 39.9727287, -46.3856926, 39.9727287, -86.3584213, 86.3584137

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1731

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1868188, upper bound: 78.1733818
time: 79.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1159069, upper bound: 78.2440180
time: 72.41 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 154.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.5179554, upper bound: 77.9409025
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.4485039, upper bound: 78.0146321
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.3894615, upper bound: 78.0167660
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.3191446, upper bound: 78.0880433
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.3089036, upper bound: 77.9938074
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.3682735, upper bound: 78.0652153
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.3089036, upper bound: 78.0663626
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.1278798, upper bound: 78.1373398
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.3840744, upper bound: 78.0209948
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.3138851, upper bound: 78.0923653
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.3128912, upper bound: 78.1533769
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.2423646, upper bound: 78.2240568
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.3038080, upper bound: 78.0765471
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.2332223, upper bound: 78.1474214
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.2320285, upper bound: 78.2066595
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.1612550, upper bound: 78.2770975
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.3608833, upper bound: 78.0119756
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.2906471, upper bound: 78.0836405
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.2396780, upper bound: 78.0846784
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.1690785, upper bound: 78.1557820
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.3244604, upper bound: 78.0947775
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.2540248, upper bound: 78.1656931
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.1988230, upper bound: 78.1667624
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.1278798, upper bound: 78.2373979
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.2232237, upper bound: 78.0910823
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.0810694, upper bound: 78.1621950
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.1519826, upper bound: 78.2219899
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.0810694, upper bound: 78.2925038
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.1868188, upper bound: 78.1733818
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 154.46
Output dim: 12, lower bound: -78.1159069, upper bound: 78.2440180
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.1223762, upper bound: 78.3828749
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.1723673, upper bound: 78.1223763
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.2538002, upper bound: 78.1970109
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.3024152, upper bound: 78.1621829
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.1723673, upper bound: 78.2331981
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.2471237, upper bound: 78.2091759
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.1756710, upper bound: 78.3344671
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.0937407, upper bound: 78.2498311
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.0937407, upper bound: 78.3706006
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.2870256, upper bound: 78.2420827
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.1577287, upper bound: 78.3135569
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.2342722, upper bound: 78.3224882
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.1026015, upper bound: 78.3934753
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.1474877, upper bound: 78.3187590
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.0753920, upper bound: 78.4480436
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.0982038, upper bound: 78.3989485
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 154.46
Output dim: 12, lower bound: -78.0247172, upper bound: 78.5274749

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 87.02 + 7204.66 = 7291.68 seconds
