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
execution time: IAR + RelationalAnalysis = 2.73 + 84.32 = 87.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -78.5698195, upper bound: 78.5698195

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.5608937, upper bound: 78.4212832
time: 71.42 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.5608937, upper bound: 78.5608934
time: 76.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 148.30 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 148.30
Output dim: 12, lower bound: -78.5608937, upper bound: 78.4212832
IS_A2, status: Status.UNKNOWN, split count: 1, time: 148.30
Output dim: 12, lower bound: -78.5608937, upper bound: 78.5608934

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -74.8727798, 52.7657051, -75.0504684, 52.9012146, -127.7739868, 127.8161774
1: -44.2210960, 40.2907715, -44.3285141, 40.3779373, -84.5990295, 84.6192856
2: -41.2945290, 41.1953583, -41.4567490, 41.3179207, -82.6124420, 82.6521072
3: -50.0107956, 45.7709618, -50.1679420, 45.9164963, -95.9272842, 95.9388962
4: -51.8464584, 48.5418549, -52.0613594, 48.6832352, -100.5296936, 100.6032104
5: -51.1129074, 49.7465515, -51.2740135, 49.9217033, -101.0346069, 101.0205688
6: -57.3771133, 41.6139374, -57.5320053, 41.7552681, -99.1323853, 99.1459427
7: -56.3844414, 45.4039001, -56.5567284, 45.5115242, -101.8959656, 101.9606323
8: -57.2151451, 53.7692375, -57.3831787, 53.9137306, -111.1288757, 111.1524200
9: -51.1587830, 47.0480042, -51.3325500, 47.2595901, -98.4183578, 98.3805389
10: -67.0643005, 59.3156128, -67.4445343, 59.7022095, -126.7665100, 126.7601395
11: -65.1131973, 45.9834938, -65.4131317, 46.2311745, -111.3443756, 111.3966217
12: -60.3829498, 60.6857719, -60.7587128, 61.0629654, -121.4459076, 121.4444733
13: -63.9668808, 56.9860344, -64.1362610, 57.1872559, -121.1541290, 121.1222992
14: -98.6686249, 48.1755219, -99.0674973, 48.5134125, -147.1820374, 147.2430115
15: -60.2210236, 43.2437325, -60.4582672, 43.4383354, -103.6593552, 103.7019882
16: -68.7277908, 46.6449165, -68.9765625, 46.8767548, -115.6045456, 115.6214676
17: -99.8998871, 53.5973358, -100.2475357, 53.8910828, -153.7909698, 153.8448639
18: -57.9793167, 49.2107620, -58.2115860, 49.3213043, -107.3006210, 107.4223404
19: -47.7861328, 34.9221725, -47.9317703, 34.9750137, -82.7611389, 82.8539429
20: -42.8983650, 36.6249886, -43.0616112, 36.7255363, -79.6239014, 79.6865997
21: -59.0447922, 40.2282448, -59.2722702, 40.3567276, -99.4015198, 99.5005188
22: -59.0003242, 40.3081055, -59.2013016, 40.4444275, -99.4447403, 99.5094070
23: -48.7998924, 40.0650177, -48.9226723, 40.1417236, -88.9416046, 88.9876862
24: -53.4456482, 39.4170837, -53.6444969, 39.5362854, -92.9819260, 93.0615768
25: -50.8949509, 45.3595581, -51.0196381, 45.4630432, -96.3579941, 96.3791885
26: -69.5211411, 61.0895538, -69.7661438, 61.2940712, -130.8152008, 130.8556976
27: -58.3800354, 40.6025314, -58.6270218, 40.7422981, -99.1223145, 99.2295532
28: -49.6633568, 44.9215126, -49.7821960, 45.0145187, -94.6778717, 94.7037048
29: -60.4086151, 35.0764771, -60.5577011, 35.2036362, -95.6122284, 95.6341782
30: -61.4312439, 44.7342758, -61.5672951, 44.8743324, -106.3055725, 106.3015594
31: -58.4951286, 39.5354004, -58.6958466, 39.5887871, -98.0839157, 98.2312469
32: -51.7899208, 39.3773842, -51.9708824, 39.5306473, -91.3205719, 91.3482666
33: -77.3161926, 49.9475861, -77.5793228, 50.1934776, -127.5096741, 127.5269089
34: -63.3392258, 42.2028351, -63.5087776, 42.4116440, -105.7508698, 105.7116089
35: -59.6315842, 42.0158386, -59.8361435, 42.2313499, -101.8629303, 101.8519821
36: -58.4734077, 42.8714104, -58.6236992, 42.9800186, -101.4534302, 101.4951096
37: -88.1156998, 50.1802597, -88.3610001, 50.3124771, -138.4281769, 138.5412598
38: -70.9741211, 52.5785027, -71.1662445, 52.7332993, -123.7074203, 123.7447357
39: -81.2956009, 44.1957855, -81.5023346, 44.3005829, -125.5961838, 125.6981201
40: -73.1659775, 39.5373802, -73.3905792, 39.6812477, -112.8472290, 112.9279633
41: -54.7032318, 43.9823456, -54.8686333, 44.0769997, -98.7802277, 98.8509827
42: -46.1607132, 39.6164742, -46.3197708, 39.8111076, -85.9718018, 85.9362488

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3974515, upper bound: 78.4066459
time: 63.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3974515, upper bound: 78.4080261
time: 67.94 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -75.1476898, 52.9449463, -75.1598434, 52.9497299, -128.0974121, 128.1047821
1: -44.3789368, 40.4032440, -44.3876343, 40.4063721, -84.7853088, 84.7908783
2: -41.5653610, 41.3423691, -41.5755005, 41.3467865, -82.9121399, 82.9178696
3: -50.2566795, 45.9590034, -50.2720718, 45.9645500, -96.2212219, 96.2310715
4: -52.2171173, 48.7217636, -52.2337112, 48.7276268, -100.9447403, 100.9554749
5: -51.3670959, 49.9660797, -51.3817596, 49.9718323, -101.3389282, 101.3478317
6: -57.5946388, 41.8236160, -57.6033669, 41.8405380, -99.4351807, 99.4269867
7: -56.6267891, 45.5476151, -56.6404266, 45.5532990, -102.1800766, 102.1880417
8: -57.4823914, 53.9569435, -57.4948196, 53.9623413, -111.4447327, 111.4517517
9: -51.3713417, 47.4115448, -51.3766212, 47.4269600, -98.7983017, 98.7881622
10: -67.5095367, 59.9988022, -67.5177460, 60.0279388, -127.5374756, 127.5165482
11: -65.4655609, 46.4152565, -65.4718704, 46.4349518, -111.9005051, 111.8871307
12: -60.8060646, 61.3474007, -60.8132858, 61.3733215, -122.1793823, 122.1606903
13: -64.1717300, 57.3032951, -64.1800995, 57.3159599, -121.4876862, 121.4833984
14: -99.1379013, 48.7806931, -99.1492615, 48.8038216, -147.9417267, 147.9299622
15: -60.6110229, 43.4868317, -60.6288757, 43.4925156, -104.1035309, 104.1157074
16: -69.0383148, 47.0361748, -69.0467072, 47.0550804, -116.0933990, 116.0828857
17: -100.2909622, 54.1114235, -100.2994537, 54.1233292, -154.4142914, 154.4108734
18: -58.3185234, 49.3639641, -58.3343468, 49.3718605, -107.6903839, 107.6983032
19: -47.9779243, 34.9948311, -47.9883156, 35.0003204, -82.9782410, 82.9831390
20: -43.1083221, 36.7919922, -43.1133118, 36.8005409, -79.9088593, 79.9053040
21: -59.3270645, 40.4357567, -59.3336716, 40.4461060, -99.7731705, 99.7694244
22: -59.2943878, 40.4948044, -59.3150902, 40.5026932, -99.7970734, 99.8098907
23: -48.9692116, 40.1740265, -48.9743996, 40.1828003, -89.1520004, 89.1484222
24: -53.7772522, 39.5558777, -53.7905731, 39.5604477, -93.3376999, 93.3464508
25: -51.0833588, 45.5036316, -51.0922966, 45.5093346, -96.5926819, 96.5959244
26: -69.8290482, 61.4284668, -69.8375397, 61.4422760, -131.2713165, 131.2659912
27: -58.7898560, 40.7624741, -58.8056412, 40.7685356, -99.5583954, 99.5681152
28: -49.8403397, 45.0418892, -49.8468208, 45.0482635, -94.8885956, 94.8887100
29: -60.6159096, 35.2690582, -60.6266289, 35.2808418, -95.8967438, 95.8956833
30: -61.6173401, 44.9604645, -61.6236649, 44.9722214, -106.5895462, 106.5841293
31: -58.7780266, 39.6094818, -58.7930794, 39.6152496, -98.3932800, 98.4025574
32: -52.0286293, 39.6366539, -52.0363159, 39.6491394, -91.6777649, 91.6729584
33: -77.7605820, 50.2290535, -77.7790375, 50.2349167, -127.9954987, 128.0080872
34: -63.6140099, 42.4448242, -63.6276131, 42.4503632, -106.0643768, 106.0724335
35: -59.9783897, 42.2630234, -59.9947128, 42.2673454, -102.2457352, 102.2577362
36: -58.7020645, 43.0049133, -58.7150192, 43.0084686, -101.7105331, 101.7199326
37: -88.4956818, 50.3505783, -88.5130844, 50.3565407, -138.8522034, 138.8636627
38: -71.2570572, 52.7655449, -71.2729416, 52.7700310, -124.0270844, 124.0384827
39: -81.5908966, 44.3248596, -81.6136017, 44.3307877, -125.9216614, 125.9384460
40: -73.5115585, 39.6996117, -73.5241241, 39.7036896, -113.2152481, 113.2237396
41: -54.9601173, 44.1133804, -54.9711456, 44.1205444, -99.0806427, 99.0845261
42: -46.3719292, 39.9263802, -46.3788490, 39.9490356, -86.3209686, 86.3052292

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3974515, upper bound: 78.5533463
time: 62.80 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3974515, upper bound: 78.5540345
time: 63.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 128.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 128.45
Output dim: 12, lower bound: -78.3974515, upper bound: 78.4066459
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 128.45
Output dim: 12, lower bound: -78.3974515, upper bound: 78.4080261
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 128.45
Output dim: 12, lower bound: -78.3974515, upper bound: 78.5533463
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 128.45
Output dim: 12, lower bound: -78.3974515, upper bound: 78.5540345

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -74.7403030, 52.7367172, -74.7364807, 52.6632195, -127.4035034, 127.4731979
1: -44.1324081, 40.2719955, -44.1208115, 40.2308083, -84.3632050, 84.3928070
2: -41.1428299, 41.1747971, -41.1237717, 41.1290588, -82.2718811, 82.2985687
3: -49.8526802, 45.7386360, -49.8230591, 45.6714096, -95.5240784, 95.5616913
4: -51.6736603, 48.5140648, -51.6783180, 48.4704781, -100.1441193, 100.1923828
5: -50.9533844, 49.7144814, -50.9191437, 49.6504822, -100.6038666, 100.6336212
6: -57.3233147, 41.5548096, -57.3551674, 41.5927391, -98.9160461, 98.9099731
7: -56.2556458, 45.3790855, -56.2401466, 45.3317642, -101.5874100, 101.6192322
8: -57.0784760, 53.7390175, -57.0737114, 53.7086792, -110.7871399, 110.8127213
9: -51.1170273, 46.9211006, -51.1590004, 46.9644699, -98.0814972, 98.0800858
10: -66.9998474, 59.0500488, -67.0566101, 59.1260643, -126.1259155, 126.1066589
11: -65.0685043, 45.8083344, -65.1225739, 45.8560867, -110.9245911, 110.9309082
12: -60.3470879, 60.3980179, -60.3287735, 60.4439430, -120.7910309, 120.7267914
13: -63.9236374, 56.9077644, -64.0099182, 56.9649353, -120.8885498, 120.9176788
14: -98.5911560, 47.9783173, -98.6573105, 48.0875969, -146.6787415, 146.6356201
15: -60.0731125, 43.1984901, -60.1185570, 43.2491455, -103.3222580, 103.3170471
16: -68.6658630, 46.5159302, -68.7387390, 46.5735626, -115.2394257, 115.2546692
17: -99.8525238, 53.4468040, -99.9193420, 53.5384636, -153.3909912, 153.3661499
18: -57.9016380, 49.1439323, -57.9430618, 49.1554947, -107.0571213, 107.0869751
19: -47.7432938, 34.8706894, -47.7550659, 34.8546371, -82.5979309, 82.6257553
20: -42.8543091, 36.5535736, -42.8681831, 36.5661049, -79.4204102, 79.4217529
21: -59.0007820, 40.1186371, -59.0194817, 40.1098557, -99.1106262, 99.1381226
22: -58.9500313, 40.2275620, -59.0318069, 40.2316437, -99.1816711, 99.2593689
23: -48.7628021, 40.0090446, -48.7710686, 40.0041962, -88.7669983, 88.7801132
24: -53.3785095, 39.3940582, -53.4650650, 39.4625664, -92.8410797, 92.8591232
25: -50.8577499, 45.2976265, -50.8942833, 45.3015518, -96.1592789, 96.1919022
26: -69.4712448, 60.8956375, -69.3963623, 60.8644142, -130.3356323, 130.2919922
27: -58.2851334, 40.5768623, -58.3834381, 40.6476059, -98.9327393, 98.9602966
28: -49.6260681, 44.8872414, -49.6507263, 44.9178047, -94.5438690, 94.5379639
29: -60.3689003, 34.9750824, -60.4014359, 34.9690857, -95.3379745, 95.3765182
30: -61.3935394, 44.6293221, -61.3867111, 44.6348610, -106.0283966, 106.0160217
31: -58.4293747, 39.4947891, -58.4591522, 39.4851189, -97.9144897, 97.9539413
32: -51.7427368, 39.2809792, -51.7828293, 39.3145294, -91.0572510, 91.0637970
33: -77.1828613, 49.9039154, -77.2728271, 49.9620514, -127.1449127, 127.1767426
34: -63.2525139, 42.1632614, -63.2975044, 42.2316475, -105.4841461, 105.4607620
35: -59.5254555, 41.9818916, -59.6001778, 42.0494308, -101.5748825, 101.5820618
36: -58.4182663, 42.8398438, -58.4737434, 42.8760757, -101.2943420, 101.3135834
37: -88.0434647, 50.1274872, -88.1411133, 50.1581192, -138.2015839, 138.2686005
38: -70.8800354, 52.5459213, -70.9146576, 52.5887413, -123.4687805, 123.4605789
39: -81.2218094, 44.1612930, -81.2918854, 44.1604843, -125.3822937, 125.4531708
40: -73.0902557, 39.5157280, -73.1679001, 39.5651627, -112.6554184, 112.6836166
41: -54.6427727, 43.9389076, -54.6988983, 43.9503326, -98.5931091, 98.6378021
42: -46.1190300, 39.4951439, -46.1506958, 39.5178719, -85.6368713, 85.6458435

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3811667, upper bound: 78.2507955
time: 67.61 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3811667, upper bound: 78.3974932
time: 95.11 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -74.8597565, 52.7626724, -75.0250854, 52.8955269, -127.7552795, 127.7877579
1: -44.2124252, 40.2886162, -44.3112679, 40.3739014, -84.5863190, 84.5998840
2: -41.2817993, 41.1927872, -41.4323463, 41.3130989, -82.5948944, 82.6251373
3: -49.9955521, 45.7669525, -50.1412544, 45.9088783, -95.9044189, 95.9082031
4: -51.8325920, 48.5376358, -52.0334320, 48.6756706, -100.5082550, 100.5710678
5: -51.1011734, 49.7426758, -51.2518005, 49.9142494, -101.0154266, 100.9944611
6: -57.3702507, 41.5990143, -57.5192757, 41.7257080, -99.0959549, 99.1182861
7: -56.3707008, 45.4007225, -56.5310631, 45.5057106, -101.8764038, 101.9317856
8: -57.2031403, 53.7655640, -57.3594627, 53.9067650, -111.1099091, 111.1250229
9: -51.1538429, 47.0365677, -51.3232803, 47.2370415, -98.3908768, 98.3598480
10: -67.0575867, 59.2924309, -67.4316940, 59.6559982, -126.7135773, 126.7241135
11: -65.1075897, 45.9682274, -65.4025269, 46.2012825, -111.3088684, 111.3707581
12: -60.3783035, 60.6625214, -60.7501068, 61.0167770, -121.3950806, 121.4126205
13: -63.9600258, 56.9763374, -64.1237411, 57.1693077, -121.1293335, 121.1000748
14: -98.6597672, 48.1599503, -99.0508499, 48.4825287, -147.1422729, 147.2108002
15: -60.2064285, 43.2381477, -60.4302368, 43.4279327, -103.6343613, 103.6683731
16: -68.7203979, 46.6314774, -68.9623642, 46.8505173, -115.5709152, 115.5938263
17: -99.8942871, 53.5862045, -100.2373352, 53.8697548, -153.7640228, 153.8235321
18: -57.9702759, 49.2014313, -58.1938820, 49.3036079, -107.2738800, 107.3953094
19: -47.7818375, 34.9176102, -47.9233055, 34.9659576, -82.7477951, 82.8409119
20: -42.8933334, 36.6181259, -43.0521545, 36.7121010, -79.6054306, 79.6702805
21: -59.0395546, 40.2189560, -59.2624016, 40.3382874, -99.3778381, 99.4813538
22: -58.9941406, 40.2964325, -59.1897049, 40.4216003, -99.4157410, 99.4861374
23: -48.7958221, 40.0568771, -48.9149132, 40.1285934, -88.9244156, 88.9717865
24: -53.4376335, 39.4132347, -53.6287842, 39.5293846, -92.9670181, 93.0420151
25: -50.8902245, 45.3528023, -51.0105171, 45.4498863, -96.3401031, 96.3633194
26: -69.5152740, 61.0706177, -69.7551117, 61.2648125, -130.7800751, 130.8257141
27: -58.3694344, 40.5985947, -58.6061783, 40.7349548, -99.1043854, 99.2047653
28: -49.6594200, 44.9155922, -49.7746048, 45.0030899, -94.6624985, 94.6901932
29: -60.4026070, 35.0652275, -60.5465698, 35.1828384, -95.5854492, 95.6117935
30: -61.4261971, 44.7247696, -61.5577354, 44.8558235, -106.2820206, 106.2825012
31: -58.4878731, 39.5301514, -58.6818008, 39.5788383, -98.0667114, 98.2119522
32: -51.7836494, 39.3703079, -51.9590988, 39.5167236, -91.3003693, 91.3293915
33: -77.3035889, 49.9426384, -77.5548172, 50.1838303, -127.4874115, 127.4974442
34: -63.3307381, 42.1980782, -63.4922867, 42.4024963, -105.7332306, 105.6903687
35: -59.6207542, 42.0124283, -59.8152046, 42.2245712, -101.8453217, 101.8276138
36: -58.4648705, 42.8677483, -58.6073380, 42.9730759, -101.4379349, 101.4750824
37: -88.1050720, 50.1746445, -88.3408585, 50.3014145, -138.4064941, 138.5155029
38: -70.9636154, 52.5745430, -71.1458282, 52.7256393, -123.6892471, 123.7203674
39: -81.2870026, 44.1918030, -81.4857788, 44.2926636, -125.5796661, 125.6775818
40: -73.1554871, 39.5338516, -73.3709869, 39.6745148, -112.8299866, 112.9048386
41: -54.6961594, 43.9734535, -54.8552628, 44.0585098, -98.7546692, 98.8287201
42: -46.1554947, 39.6001816, -46.3102303, 39.7844582, -85.9399567, 85.9104156

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3811667, upper bound: 78.2552622
time: 78.15 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3811667, upper bound: 78.3990172
time: 77.73 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -75.0147171, 52.9158745, -74.8454132, 52.7120743, -127.7267914, 127.7612915
1: -44.2903252, 40.3843994, -44.1800728, 40.2594070, -84.5497284, 84.5644684
2: -41.4134712, 41.3218689, -41.2428436, 41.1580658, -82.5715332, 82.5647125
3: -50.0960007, 45.9264336, -49.9254494, 45.7195816, -95.8155823, 95.8518753
4: -52.0439682, 48.6934891, -51.8507004, 48.5146332, -100.5586014, 100.5441895
5: -51.2035217, 49.9340706, -51.0207253, 49.7007332, -100.9042511, 100.9547958
6: -57.5406075, 41.7632446, -57.4265556, 41.6779900, -99.2185822, 99.1898041
7: -56.4969940, 45.5224113, -56.3231087, 45.3736496, -101.8706436, 101.8455200
8: -57.3457832, 53.9266968, -57.1852112, 53.7574768, -111.1032562, 111.1119080
9: -51.3296318, 47.2844086, -51.2030144, 47.1317673, -98.4613953, 98.4874268
10: -67.4455261, 59.7334175, -67.1302338, 59.4516678, -126.8971863, 126.8636475
11: -65.4210663, 46.2397385, -65.1817780, 46.0594864, -111.4805527, 111.4215164
12: -60.7697601, 61.0598030, -60.3829002, 60.7543373, -121.5240936, 121.4427032
13: -64.1277847, 57.2241287, -64.0532455, 57.0931892, -121.2209778, 121.2773590
14: -99.0600052, 48.5831985, -98.7391205, 48.3777275, -147.4377289, 147.3223114
15: -60.4621658, 43.4425125, -60.2881126, 43.3035469, -103.7657166, 103.7306213
16: -68.9766769, 46.9064102, -68.8087082, 46.7510948, -115.7277603, 115.7151184
17: -100.2431335, 53.9563637, -99.9708939, 53.7688179, -154.0119476, 153.9272614
18: -58.2390671, 49.2968559, -58.0652046, 49.2056427, -107.4447098, 107.3620605
19: -47.9348984, 34.9437447, -47.8128815, 34.8796768, -82.8145752, 82.7566223
20: -43.0639648, 36.7203484, -42.9203110, 36.6408539, -79.7048187, 79.6406555
21: -59.2828712, 40.3261871, -59.0807076, 40.1990738, -99.4819336, 99.4068909
22: -59.2428474, 40.4141808, -59.1450233, 40.2892838, -99.5321350, 99.5592041
23: -48.9317207, 40.1153603, -48.8228111, 40.0428619, -88.9745789, 88.9381714
24: -53.7092018, 39.5323296, -53.6102905, 39.4862595, -93.1954575, 93.1426239
25: -51.0453110, 45.4422493, -50.9678879, 45.3474846, -96.3927917, 96.4101410
26: -69.7787170, 61.2266350, -69.4681702, 61.0034065, -130.7821198, 130.6948090
27: -58.6944618, 40.7367668, -58.5616379, 40.6737976, -99.3682556, 99.2984009
28: -49.8026276, 45.0073509, -49.7150459, 44.9511108, -94.7537384, 94.7223969
29: -60.5749893, 35.1666603, -60.4703026, 35.0453415, -95.6203308, 95.6369629
30: -61.5801926, 44.8549995, -61.4441223, 44.7321701, -106.3123627, 106.2991180
31: -58.7116547, 39.5685196, -58.5558548, 39.5112228, -98.2228699, 98.1243668
32: -51.9813652, 39.5400925, -51.8483925, 39.4330101, -91.4143753, 91.3884888
33: -77.6265182, 50.1853371, -77.4719238, 50.0036621, -127.6301727, 127.6572571
34: -63.5268173, 42.4051666, -63.4158401, 42.2706604, -105.7974777, 105.8210068
35: -59.8716278, 42.2294235, -59.7582436, 42.0859184, -101.9575500, 101.9876633
36: -58.6460457, 42.9738922, -58.5643539, 42.9043770, -101.5504227, 101.5382462
37: -88.4211349, 50.2980804, -88.2919922, 50.2023087, -138.6234436, 138.5900574
38: -71.1615067, 52.7331085, -71.0201721, 52.6256981, -123.7871933, 123.7532806
39: -81.5163803, 44.2905273, -81.4028397, 44.1908379, -125.7072144, 125.6933670
40: -73.4345093, 39.6776466, -73.2999268, 39.5875664, -113.0220642, 112.9775696
41: -54.8987160, 44.0705795, -54.8007164, 43.9940491, -98.8927612, 98.8712921
42: -46.3302193, 39.7969437, -46.2099342, 39.6501427, -85.9803391, 86.0068817

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3816231, upper bound: 78.4007769
time: 65.11 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3876149, upper bound: 78.5433247
time: 72.87 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -75.1348648, 52.9421310, -75.1358490, 52.9444962, -128.0793610, 128.0779724
1: -44.3701248, 40.4012909, -44.3706779, 40.4027634, -84.7728882, 84.7719650
2: -41.5555038, 41.3399582, -41.5551491, 41.3422546, -82.8977585, 82.8951111
3: -50.2444763, 45.9553566, -50.2489586, 45.9576225, -96.2021027, 96.2043152
4: -52.2029686, 48.7179527, -52.2063904, 48.7203827, -100.9233551, 100.9243469
5: -51.3556480, 49.9624634, -51.3599815, 49.9648781, -101.3205261, 101.3224487
6: -57.5881615, 41.8087387, -57.5909576, 41.8113251, -99.3994751, 99.3996964
7: -56.6128235, 45.5448036, -56.6153488, 45.5480423, -102.1608658, 102.1601562
8: -57.4702339, 53.9535255, -57.4715233, 53.9558449, -111.4260712, 111.4250336
9: -51.3667679, 47.4001312, -51.3680038, 47.4049377, -98.7717056, 98.7681351
10: -67.5029755, 59.9752693, -67.5053558, 59.9822731, -127.4852448, 127.4806213
11: -65.4601288, 46.4002151, -65.4616547, 46.4059868, -111.8661041, 111.8618698
12: -60.8018227, 61.3243027, -60.8051147, 61.3278885, -122.1296997, 122.1294174
13: -64.1655884, 57.2940559, -64.1688385, 57.2982101, -121.4637985, 121.4628906
14: -99.1296616, 48.7690430, -99.1334915, 48.7753143, -147.9049683, 147.9025269
15: -60.5969887, 43.4815063, -60.6026306, 43.4824371, -104.0794220, 104.0841370
16: -69.0312805, 47.0231247, -69.0331421, 47.0304108, -116.0616913, 116.0562668
17: -100.2859650, 54.1007729, -100.2899857, 54.1025734, -154.3885345, 154.3907471
18: -58.3097610, 49.3548355, -58.3172379, 49.3546600, -107.6644211, 107.6720734
19: -47.9737015, 34.9901199, -47.9801636, 34.9915161, -82.9652176, 82.9702835
20: -43.1036110, 36.7852554, -43.1043701, 36.7874603, -79.8910675, 79.8896255
21: -59.3220520, 40.4263687, -59.3240623, 40.4280472, -99.7500992, 99.7504272
22: -59.2886620, 40.4832001, -59.3043137, 40.4800034, -99.7686615, 99.7875137
23: -48.9652901, 40.1673660, -48.9669113, 40.1704636, -89.1357574, 89.1342697
24: -53.7700577, 39.5523758, -53.7755775, 39.5540237, -93.3240814, 93.3279572
25: -51.0789108, 45.4968948, -51.0837936, 45.4966545, -96.5755615, 96.5806885
26: -69.8237076, 61.4139023, -69.8275299, 61.4141922, -131.2378998, 131.2414246
27: -58.7793770, 40.7587166, -58.7851486, 40.7615318, -99.5409088, 99.5438538
28: -49.8365326, 45.0360794, -49.8395233, 45.0373459, -94.8738785, 94.8756027
29: -60.6104736, 35.2578316, -60.6163101, 35.2603874, -95.8708572, 95.8741455
30: -61.6126137, 44.9512329, -61.6145439, 44.9545212, -106.5671234, 106.5657806
31: -58.7708817, 39.6044312, -58.7792664, 39.6060257, -98.3769073, 98.3836975
32: -52.0227127, 39.6307526, -52.0249214, 39.6368179, -91.6595306, 91.6556702
33: -77.7483215, 50.2241707, -77.7551727, 50.2256775, -127.9739838, 127.9793396
34: -63.6055794, 42.4402161, -63.6113205, 42.4418945, -106.0474548, 106.0515366
35: -59.9677925, 42.2594604, -59.9742928, 42.2605972, -102.2283936, 102.2337494
36: -58.6938934, 43.0013199, -58.6993675, 43.0017433, -101.6956329, 101.7006836
37: -88.4859161, 50.3448257, -88.4941635, 50.3457642, -138.8316803, 138.8389893
38: -71.2466507, 52.7616844, -71.2531586, 52.7627220, -124.0093689, 124.0148392
39: -81.5827866, 44.3206940, -81.5981064, 44.3229561, -125.9057388, 125.9188004
40: -73.5019073, 39.6962090, -73.5052338, 39.6972656, -113.1991730, 113.2014465
41: -54.9533615, 44.1039085, -54.9581108, 44.1021729, -99.0555344, 99.0620193
42: -46.3669968, 39.9115753, -46.3695602, 39.9232635, -86.2902603, 86.2811356

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3816231, upper bound: 78.4027267
time: 68.58 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3811667, upper bound: 78.5440441
time: 73.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 144.36 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 144.36
Output dim: 12, lower bound: -78.3811667, upper bound: 78.2507955
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 144.36
Output dim: 12, lower bound: -78.3811667, upper bound: 78.3974932
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 144.36
Output dim: 12, lower bound: -78.3811667, upper bound: 78.2552622
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 144.36
Output dim: 12, lower bound: -78.3811667, upper bound: 78.3990172
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 144.36
Output dim: 12, lower bound: -78.3816231, upper bound: 78.4007769
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 144.36
Output dim: 12, lower bound: -78.3876149, upper bound: 78.5433247
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 144.36
Output dim: 12, lower bound: -78.3816231, upper bound: 78.4027267
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 144.36
Output dim: 12, lower bound: -78.3811667, upper bound: 78.5440441

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -74.4591293, 52.5735245, -74.6108856, 52.6235962, -127.0827179, 127.1844101
1: -43.9661064, 40.1758118, -44.0472145, 40.2068100, -84.1729126, 84.2230225
2: -40.8531952, 41.0248108, -40.9846878, 41.1042328, -81.9574203, 82.0094986
3: -49.5590172, 45.5614395, -49.6814079, 45.6284676, -95.1874847, 95.2428360
4: -51.3107376, 48.3655853, -51.5029755, 48.4353104, -99.7460480, 99.8685608
5: -50.6574402, 49.4876480, -50.7758522, 49.6049576, -100.2623978, 100.2634964
6: -57.1510239, 41.3121643, -57.2987480, 41.4808502, -98.6318665, 98.6109085
7: -55.9838448, 45.2541351, -56.1205711, 45.2964592, -101.2803040, 101.3747101
8: -56.8169594, 53.5702782, -56.9527779, 53.6697960, -110.4867554, 110.5230484
9: -50.9876938, 46.5980301, -51.1138153, 46.8183899, -97.8060760, 97.7118454
10: -66.7459412, 58.5063400, -66.9915466, 58.8727913, -125.6187286, 125.4978867
11: -64.8337097, 45.4312286, -65.0657806, 45.6745987, -110.5082855, 110.4970093
12: -59.9847260, 59.7021217, -60.2800293, 60.1050453, -120.0897675, 119.9821472
13: -63.7454453, 56.6364136, -63.9633026, 56.8496361, -120.5950775, 120.5997162
14: -98.2239609, 47.4538155, -98.5681686, 47.8326912, -146.0566559, 146.0219727
15: -59.7072868, 43.0122681, -59.9464111, 43.2021294, -102.9094162, 102.9586792
16: -68.4479828, 46.1996689, -68.6688843, 46.4302368, -114.8782196, 114.8685455
17: -99.5136642, 52.9340134, -99.8588867, 53.2925529, -152.8062134, 152.7928925
18: -57.6223259, 48.9877663, -57.8293457, 49.0929871, -106.7153168, 106.8171082
19: -47.5710754, 34.7816162, -47.6983795, 34.8156548, -82.3867340, 82.4799957
20: -42.6838722, 36.3751144, -42.8181953, 36.4839859, -79.1678467, 79.1933136
21: -58.7862244, 39.9002953, -58.9633408, 40.0102844, -98.7965012, 98.8636322
22: -58.7309799, 40.0248947, -58.9461021, 40.1492081, -98.8801880, 98.9709930
23: -48.6030273, 39.9081955, -48.7178307, 39.9631042, -88.5661316, 88.6260223
24: -53.0783463, 39.2808228, -53.3295593, 39.4386559, -92.5169983, 92.6103821
25: -50.6631889, 45.1748352, -50.8156548, 45.2562599, -95.9194336, 95.9904938
26: -69.1928101, 60.5121918, -69.3310013, 60.6837120, -129.8765106, 129.8432007
27: -57.9606781, 40.4813881, -58.2385788, 40.6230469, -98.5837250, 98.7199554
28: -49.4825630, 44.8134384, -49.6044846, 44.8859482, -94.3685150, 94.4179230
29: -60.2079048, 34.7320061, -60.3455925, 34.8547363, -95.0626373, 95.0775986
30: -61.2394676, 44.4296112, -61.3336411, 44.5454712, -105.7849426, 105.7632523
31: -58.1591835, 39.4211617, -58.3549576, 39.4561424, -97.6153259, 97.7761154
32: -51.5287514, 38.9894104, -51.7291603, 39.1743813, -90.7031250, 90.7185669
33: -76.7758942, 49.7123604, -77.0949478, 49.9162445, -126.6921387, 126.8073044
34: -63.0418816, 42.0095444, -63.2100716, 42.1936531, -105.2355270, 105.2196198
35: -59.2586212, 41.8530922, -59.4848938, 42.0166550, -101.2752686, 101.3379822
36: -58.2681313, 42.7408600, -58.4257011, 42.8362427, -101.1043701, 101.1665573
37: -87.7557144, 50.0037918, -88.0308685, 50.1098366, -137.8655396, 138.0346680
38: -70.6276245, 52.4297791, -70.8318787, 52.5490646, -123.1766815, 123.2616501
39: -80.9551315, 44.0543327, -81.1948700, 44.1290779, -125.0842133, 125.2492065
40: -72.8257523, 39.4152679, -73.0616913, 39.5399017, -112.3656464, 112.4769592
41: -54.4618683, 43.7854271, -54.6290703, 43.8847275, -98.3465958, 98.4144897
42: -45.9496803, 39.1824646, -46.1050949, 39.3709030, -85.3205872, 85.2875443

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2646196, upper bound: 78.2444293
time: 87.65 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2646196, upper bound: 78.2444898
time: 79.45 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -74.7213593, 52.7307816, -74.7265396, 52.6603317, -127.3816910, 127.4573212
1: -44.1187477, 40.2680206, -44.1140404, 40.2288513, -84.3475952, 84.3820496
2: -41.1230507, 41.1700554, -41.1141129, 41.1267967, -82.2498322, 82.2841644
3: -49.8327827, 45.7317772, -49.8128204, 45.6680107, -95.5007935, 95.5445938
4: -51.6478195, 48.5072823, -51.6660385, 48.4671135, -100.1149292, 100.1733170
5: -50.9370041, 49.7075691, -50.9108353, 49.6470871, -100.5840759, 100.6184082
6: -57.3140030, 41.5335350, -57.3506432, 41.5831070, -98.8970947, 98.8841705
7: -56.2346573, 45.3720436, -56.2293472, 45.3283005, -101.5629425, 101.6013947
8: -57.0586891, 53.7322731, -57.0642319, 53.7053452, -110.7640228, 110.7965088
9: -51.1102104, 46.8997078, -51.1555061, 46.9543571, -98.0645599, 98.0552139
10: -66.9879074, 59.0149117, -67.0508728, 59.1094742, -126.0973816, 126.0657806
11: -65.0592957, 45.7812920, -65.1181030, 45.8431358, -110.9024200, 110.8993988
12: -60.3383217, 60.3551712, -60.3244705, 60.4232826, -120.7616043, 120.6796417
13: -63.9137726, 56.8832169, -64.0049438, 56.9531708, -120.8669434, 120.8881531
14: -98.5745087, 47.9483795, -98.6491165, 48.0732880, -146.6477966, 146.5975037
15: -60.0446663, 43.1905632, -60.1046906, 43.2453690, -103.2900238, 103.2952576
16: -68.6549072, 46.4904099, -68.7332764, 46.5613861, -115.2162628, 115.2236862
17: -99.8411331, 53.4160042, -99.9139099, 53.5236168, -153.3647461, 153.3299103
18: -57.8794823, 49.1267319, -57.9322090, 49.1471252, -107.0266113, 107.0589371
19: -47.7316742, 34.8616333, -47.7493591, 34.8501663, -82.5818253, 82.6109924
20: -42.8470688, 36.5406456, -42.8644905, 36.5598831, -79.4069519, 79.4051361
21: -58.9916000, 40.1034470, -59.0149803, 40.1024704, -99.0940704, 99.1184235
22: -58.9248085, 40.2078781, -59.0200310, 40.2221222, -99.1469269, 99.2279053
23: -48.7511673, 39.9986877, -48.7653694, 39.9991531, -88.7503128, 88.7640533
24: -53.3571320, 39.3892822, -53.4548912, 39.4600906, -92.8171997, 92.8441620
25: -50.8428192, 45.2896156, -50.8869667, 45.2976036, -96.1404266, 96.1765823
26: -69.4594193, 60.8702354, -69.3906403, 60.8520432, -130.3114471, 130.2608795
27: -58.2616768, 40.5700150, -58.3717499, 40.6443710, -98.9060440, 98.9417648
28: -49.6198006, 44.8785896, -49.6475716, 44.9134903, -94.5332947, 94.5261612
29: -60.3572311, 34.9646454, -60.3956795, 34.9639282, -95.3211594, 95.3603210
30: -61.3833466, 44.6108017, -61.3817215, 44.6258125, -106.0091553, 105.9925232
31: -58.4094849, 39.4871445, -58.4488945, 39.4812469, -97.8907318, 97.9360352
32: -51.7336578, 39.2594872, -51.7783470, 39.3043480, -91.0380020, 91.0378342
33: -77.1586456, 49.8958206, -77.2612305, 49.9580193, -127.1166687, 127.1570511
34: -63.2379913, 42.1548500, -63.2905426, 42.2275124, -105.4654999, 105.4453812
35: -59.5072632, 41.9757843, -59.5914993, 42.0465279, -101.5537872, 101.5672836
36: -58.4099388, 42.8276672, -58.4696541, 42.8702316, -101.2801666, 101.2973175
37: -88.0186386, 50.1175308, -88.1290741, 50.1531448, -138.1717834, 138.2466125
38: -70.8657074, 52.5355263, -70.9076767, 52.5836372, -123.4493408, 123.4432068
39: -81.2058563, 44.1548386, -81.2841110, 44.1571426, -125.3629990, 125.4389496
40: -73.0730133, 39.5092506, -73.1595459, 39.5620346, -112.6350479, 112.6687927
41: -54.6324768, 43.9267578, -54.6938972, 43.9445648, -98.5770416, 98.6206512
42: -46.1119995, 39.4679146, -46.1472511, 39.5048141, -85.6167908, 85.6151581

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3897204
time: 88.27 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3798320, upper bound: 78.3897204
time: 75.93 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -74.5776978, 52.5995483, -74.8982773, 52.8559341, -127.4336243, 127.4978256
1: -44.0447006, 40.1924286, -44.2358017, 40.3498344, -84.3945312, 84.4282303
2: -40.9920692, 41.0426559, -41.2929688, 41.2881622, -82.2802277, 82.3356094
3: -49.7017212, 45.5893936, -49.9991302, 45.8657455, -95.5674667, 95.5885162
4: -51.4688797, 48.3891144, -51.8570671, 48.6403885, -100.1092682, 100.2461777
5: -50.8050575, 49.5156937, -51.1081429, 49.8685989, -100.6736526, 100.6238327
6: -57.1973495, 41.3547592, -57.4622803, 41.6113930, -98.8087463, 98.8170395
7: -56.0978966, 45.2755051, -56.4104004, 45.4703445, -101.5682373, 101.6859055
8: -56.9407578, 53.5965233, -57.2373352, 53.8676376, -110.8083954, 110.8338394
9: -51.0248070, 46.7130966, -51.2783089, 47.0905380, -98.1153336, 97.9913940
10: -66.8036575, 58.7485580, -67.3664398, 59.4022446, -126.2058868, 126.1149902
11: -64.8724060, 45.5910263, -65.3453674, 46.0194283, -110.8918304, 110.9363861
12: -60.0155945, 59.9663811, -60.7009926, 60.6774445, -120.6930389, 120.6673737
13: -63.7812614, 56.7019119, -64.0765762, 57.0509872, -120.8322449, 120.7784882
14: -98.2922440, 47.6347809, -98.9614105, 48.2268906, -146.5191345, 146.5961914
15: -59.8400116, 43.0505409, -60.2569427, 43.3798409, -103.2198410, 103.3074799
16: -68.5011673, 46.3142014, -68.8911362, 46.7059479, -115.2071152, 115.2053375
17: -99.5551071, 53.0725822, -100.1766281, 53.6230469, -153.1781311, 153.2492065
18: -57.6870270, 49.0449371, -58.0771751, 49.2405777, -106.9275818, 107.1221161
19: -47.6088600, 34.8284073, -47.8664551, 34.9265671, -82.5354309, 82.6948624
20: -42.7226601, 36.4394646, -43.0022430, 36.6296997, -79.3523560, 79.4417114
21: -58.8246269, 40.0004807, -59.2057610, 40.2384033, -99.0630341, 99.2062378
22: -58.7746086, 40.0918541, -59.1035194, 40.3373184, -99.1119232, 99.1953735
23: -48.6354980, 39.9557724, -48.8613358, 40.0870285, -88.7225189, 88.8171005
24: -53.1355400, 39.2999496, -53.4913483, 39.5053635, -92.6408920, 92.7912979
25: -50.6951065, 45.2290001, -50.9318352, 45.4033508, -96.0984573, 96.1608353
26: -69.2365875, 60.6869049, -69.6896362, 61.0834122, -130.3200073, 130.3765411
27: -58.0429230, 40.5029106, -58.4592667, 40.7100677, -98.7529907, 98.9621735
28: -49.5153351, 44.8413773, -49.7277489, 44.9707489, -94.4860840, 94.5691223
29: -60.2412949, 34.8208542, -60.4905510, 35.0674744, -95.3087616, 95.3114014
30: -61.2713623, 44.5244446, -61.5048599, 44.7660675, -106.0374298, 106.0293045
31: -58.2157173, 39.4562073, -58.5759277, 39.5494156, -97.7651367, 98.0321350
32: -51.5695953, 39.0783615, -51.9054718, 39.3762627, -90.9458389, 90.9838333
33: -76.8959961, 49.7509766, -77.3762054, 50.1380920, -127.0340881, 127.1271820
34: -63.1194649, 42.0439186, -63.4040833, 42.3642273, -105.4836884, 105.4479980
35: -59.3532791, 41.8836288, -59.6991196, 42.1917343, -101.5449982, 101.5827408
36: -58.3143349, 42.7679901, -58.5587921, 42.9326019, -101.2469254, 101.3267822
37: -87.8152542, 50.0508728, -88.2285461, 50.2531204, -138.0683746, 138.2794189
38: -70.7111130, 52.4581604, -71.0617065, 52.6859741, -123.3970795, 123.5198669
39: -81.0191879, 44.0847397, -81.3875046, 44.2612228, -125.2804108, 125.4722366
40: -72.8886566, 39.4330750, -73.2624664, 39.6489487, -112.5376053, 112.6955414
41: -54.5140953, 43.8197556, -54.7839279, 43.9927788, -98.5068741, 98.6036835
42: -45.9862709, 39.2859840, -46.2647591, 39.6357727, -85.6220398, 85.5507431

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.4254198, upper bound: 78.2492643
time: 74.72 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2646196, upper bound: 78.2492643
time: 92.43 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -74.8408508, 52.7567215, -75.0151825, 52.8925858, -127.7334290, 127.7719040
1: -44.1988411, 40.2845726, -44.3045158, 40.3718796, -84.5707169, 84.5890884
2: -41.2620354, 41.1880264, -41.4227448, 41.3107452, -82.5727768, 82.6107712
3: -49.9757042, 45.7600899, -50.1310158, 45.9054527, -95.8811493, 95.8911057
4: -51.8067131, 48.5309258, -52.0212364, 48.6723480, -100.4790649, 100.5521545
5: -51.0848236, 49.7357712, -51.2436066, 49.9108276, -100.9956436, 100.9793777
6: -57.3609924, 41.5783920, -57.5147285, 41.7160950, -99.0770874, 99.0931091
7: -56.3486137, 45.3936386, -56.5189133, 45.5021667, -101.8507767, 101.9125443
8: -57.1833992, 53.7588387, -57.3501015, 53.9033890, -111.0867920, 111.1089325
9: -51.1470337, 47.0151329, -51.3197441, 47.2270012, -98.3740234, 98.3348770
10: -67.0457001, 59.2572823, -67.4259186, 59.6394882, -126.6851807, 126.6831970
11: -65.0983887, 45.9411240, -65.3981018, 46.1882401, -111.2866211, 111.3392258
12: -60.3695259, 60.6196671, -60.7457008, 60.9961395, -121.3656464, 121.3653564
13: -63.9499817, 56.9515991, -64.1185455, 57.1575394, -121.1075134, 121.0701447
14: -98.6430588, 48.1300201, -99.0426025, 48.4682007, -147.1112518, 147.1726227
15: -60.1777382, 43.2302551, -60.4160423, 43.4242020, -103.6019440, 103.6463013
16: -68.7094498, 46.6057701, -68.9569092, 46.8381920, -115.5476379, 115.5626755
17: -99.8829041, 53.5553513, -100.2318954, 53.8548088, -153.7377014, 153.7872314
18: -57.9480362, 49.1841125, -58.1831169, 49.2951012, -107.2431335, 107.3672180
19: -47.7699623, 34.9085312, -47.9175148, 34.9614716, -82.7314301, 82.8260498
20: -42.8860054, 36.6052246, -43.0484123, 36.7058640, -79.5918655, 79.6536407
21: -59.0304222, 40.2037582, -59.2579346, 40.3309250, -99.3613434, 99.4616852
22: -58.9689445, 40.2766800, -59.1778221, 40.4121017, -99.3810425, 99.4544983
23: -48.7841072, 40.0464630, -48.9092255, 40.1235428, -88.9076385, 88.9556885
24: -53.4161873, 39.4084549, -53.6186562, 39.5268250, -92.9430084, 93.0270920
25: -50.8753090, 45.3446503, -51.0032730, 45.4458389, -96.3211365, 96.3479233
26: -69.5033417, 61.0452385, -69.7492523, 61.2523613, -130.7557068, 130.7944946
27: -58.3460541, 40.5917664, -58.5945168, 40.7316475, -99.0776978, 99.1862793
28: -49.6530685, 44.9068413, -49.7713890, 44.9987259, -94.6517944, 94.6782303
29: -60.3908730, 35.0538597, -60.5406952, 35.1764984, -95.5673676, 95.5945587
30: -61.4160004, 44.7061768, -61.5528564, 44.8467484, -106.2627487, 106.2590256
31: -58.4680214, 39.5224304, -58.6715469, 39.5748138, -98.0428314, 98.1939774
32: -51.7746048, 39.3488350, -51.9546127, 39.5065918, -91.2811890, 91.3034439
33: -77.2793121, 49.9344406, -77.5431519, 50.1797180, -127.4590302, 127.4775696
34: -63.3162003, 42.1896210, -63.4852333, 42.3982620, -105.7144623, 105.6748505
35: -59.6025352, 42.0063057, -59.8064041, 42.2216644, -101.8241882, 101.8127060
36: -58.4564857, 42.8554955, -58.6030998, 42.9673309, -101.4238052, 101.4585953
37: -88.0801086, 50.1647110, -88.3286285, 50.2964325, -138.3765259, 138.4933472
38: -70.9491882, 52.5641060, -71.1387558, 52.7205200, -123.6696930, 123.7028656
39: -81.2710114, 44.1852798, -81.4779510, 44.2893600, -125.5603714, 125.6632309
40: -73.1382294, 39.5272560, -73.3625488, 39.6712952, -112.8095245, 112.8898010
41: -54.6859245, 43.9612656, -54.8502197, 44.0527382, -98.7386627, 98.8114853
42: -46.1485023, 39.5730820, -46.3068199, 39.7712860, -85.9197845, 85.8798981

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3914811
time: 82.59 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3914811
time: 66.41 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -74.7286072, 52.7545166, -74.7192841, 52.6726837, -127.4012909, 127.4737930
1: -44.1195793, 40.2892494, -44.1052628, 40.2355614, -84.3551331, 84.3945160
2: -41.1187630, 41.1725464, -41.1026306, 41.1332817, -82.2520370, 82.2751770
3: -49.8006630, 45.7493896, -49.7842445, 45.6768723, -95.4775391, 95.5336304
4: -51.6734009, 48.5438690, -51.6738968, 48.4795074, -100.1529007, 100.2177658
5: -50.9033661, 49.7058220, -50.8767014, 49.6553535, -100.5587158, 100.5825195
6: -57.3644676, 41.5072327, -57.3701210, 41.5622139, -98.9266815, 98.8773499
7: -56.2210388, 45.3979721, -56.2027245, 45.3384819, -101.5594940, 101.6006927
8: -57.0776787, 53.7576828, -57.0630455, 53.7187805, -110.7964630, 110.8207245
9: -51.2002754, 46.9558182, -51.1580544, 46.9847946, -98.1850739, 98.1138763
10: -67.1905746, 59.1809959, -67.0650024, 59.1966820, -126.3872528, 126.2460022
11: -65.1855011, 45.8564339, -65.1249695, 45.8769417, -111.0624390, 110.9813995
12: -60.4085083, 60.3576088, -60.3342743, 60.4142723, -120.8227692, 120.6918793
13: -63.9485435, 56.9467010, -64.0066528, 56.9767418, -120.9252853, 120.9533539
14: -98.6923828, 48.0567932, -98.6499786, 48.1224213, -146.8148041, 146.7067719
15: -60.0897293, 43.2512550, -60.1149597, 43.2565308, -103.3462524, 103.3662109
16: -68.7529831, 46.5832710, -68.7387848, 46.6064453, -115.3594208, 115.3220520
17: -99.9032364, 53.4400558, -99.9103775, 53.5223579, -153.4255829, 153.3504333
18: -57.9528542, 49.1383553, -57.9503708, 49.1429405, -107.0957794, 107.0887222
19: -47.7603264, 34.8524933, -47.7563210, 34.8403969, -82.6007080, 82.6088104
20: -42.8937149, 36.5395737, -42.8701096, 36.5584297, -79.4521484, 79.4096832
21: -59.0676231, 40.1060028, -59.0245247, 40.0991974, -99.1668167, 99.1305084
22: -59.0182762, 40.2067299, -59.0577545, 40.2065201, -99.2247925, 99.2644806
23: -48.7702560, 40.0119705, -48.7695312, 40.0015526, -88.7717972, 88.7815018
24: -53.4018517, 39.4180450, -53.4738159, 39.4623375, -92.8641815, 92.8918610
25: -50.8473091, 45.3167152, -50.8886566, 45.3018532, -96.1491623, 96.2053680
26: -69.4971237, 60.8440971, -69.4029388, 60.8236656, -130.3207703, 130.2470398
27: -58.3631287, 40.6373405, -58.4156113, 40.6490479, -99.0121613, 99.0529480
28: -49.6573105, 44.9313965, -49.6681213, 44.9192123, -94.5765228, 94.5995102
29: -60.4116325, 34.9221115, -60.4141006, 34.9311028, -95.3427124, 95.3362122
30: -61.4237099, 44.6504021, -61.3909912, 44.6421623, -106.0658646, 106.0413895
31: -58.4348106, 39.4934158, -58.4504738, 39.4822540, -97.9170685, 97.9438934
32: -51.7655487, 39.2429657, -51.7946930, 39.2921906, -91.0577393, 91.0376587
33: -77.2150421, 49.9935188, -77.2932816, 49.9579926, -127.1730347, 127.2867966
34: -63.3129501, 42.2502823, -63.3278313, 42.2326889, -105.5456314, 105.5781097
35: -59.5999680, 42.0998497, -59.6421432, 42.0531731, -101.6531372, 101.7419891
36: -58.4941521, 42.8718109, -58.5161705, 42.8644104, -101.3585663, 101.3879852
37: -88.1275787, 50.1719589, -88.1805573, 50.1540413, -138.2816162, 138.3525085
38: -70.9038849, 52.6158180, -70.9363708, 52.5859375, -123.4898224, 123.5521851
39: -81.2449112, 44.1823654, -81.3052979, 44.1594925, -125.4044037, 125.4876633
40: -73.1665344, 39.5760918, -73.1930695, 39.5624695, -112.7290039, 112.7691498
41: -54.7141113, 43.9138412, -54.7306519, 43.9279938, -98.6421051, 98.6444931
42: -46.1587944, 39.4728966, -46.1643600, 39.4995766, -85.6583710, 85.6372528

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3946298
time: 73.63 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3946298
time: 71.78 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -74.9943466, 52.9100151, -74.8344116, 52.7091713, -127.7035065, 127.7444305
1: -44.2757530, 40.3804321, -44.1725922, 40.2574768, -84.5332336, 84.5530167
2: -41.3937759, 41.3172798, -41.2331848, 41.1558228, -82.5495987, 82.5504608
3: -50.0764198, 45.9196243, -49.9152260, 45.7162704, -95.7926941, 95.8348465
4: -52.0191307, 48.6868935, -51.8383522, 48.5113602, -100.5304871, 100.5252457
5: -51.1844215, 49.9271965, -51.0104904, 49.6973915, -100.8818054, 100.9376755
6: -57.5317345, 41.7443466, -57.4220963, 41.6686478, -99.2003784, 99.1664429
7: -56.4759636, 45.5154343, -56.3123550, 45.3702240, -101.8461761, 101.8277740
8: -57.3267746, 53.9200287, -57.1757812, 53.7542152, -111.0809784, 111.0958023
9: -51.3227310, 47.2637787, -51.1995888, 47.1217270, -98.4444580, 98.4633636
10: -67.4340210, 59.6993179, -67.1245422, 59.4350204, -126.8690414, 126.8238602
11: -65.4120483, 46.2132454, -65.1773071, 46.0464020, -111.4584351, 111.3905487
12: -60.7611313, 61.0170403, -60.3786774, 60.7334938, -121.4946289, 121.3957138
13: -64.1177139, 57.2000237, -64.0483551, 57.0813751, -121.1990814, 121.2483826
14: -99.0435028, 48.5533829, -98.7310257, 48.3634033, -147.4069061, 147.2844086
15: -60.4336472, 43.4350090, -60.2741776, 43.2998390, -103.7334900, 103.7091751
16: -68.9658737, 46.8814087, -68.8033600, 46.7388840, -115.7047577, 115.6847687
17: -100.2322464, 53.9255905, -99.9655304, 53.7539673, -153.9862061, 153.8911133
18: -58.2174988, 49.2796898, -58.0546074, 49.1973076, -107.4148102, 107.3342972
19: -47.9229088, 34.9348183, -47.8069267, 34.8752670, -82.7981720, 82.7417450
20: -43.0564308, 36.7076950, -42.9165993, 36.6346512, -79.6910782, 79.6242905
21: -59.2738571, 40.3110580, -59.0762215, 40.1917648, -99.4656219, 99.3872833
22: -59.2187157, 40.3947792, -59.1331291, 40.2797737, -99.4984818, 99.5279083
23: -48.9199295, 40.1052780, -48.8170967, 40.0378647, -88.9577789, 88.9223633
24: -53.6883354, 39.5276222, -53.6000481, 39.4838486, -93.1721802, 93.1276627
25: -51.0303268, 45.4343605, -50.9605484, 45.3435707, -96.3739014, 96.3948898
26: -69.7670288, 61.2011299, -69.4624786, 60.9910583, -130.7580872, 130.6636047
27: -58.6715927, 40.7303467, -58.5501289, 40.6705780, -99.3421631, 99.2804718
28: -49.7959671, 44.9990234, -49.7117653, 44.9469032, -94.7428589, 94.7107849
29: -60.5633278, 35.1556244, -60.4644890, 35.0395737, -95.6028900, 95.6201172
30: -61.5702286, 44.8365746, -61.4392014, 44.7231255, -106.2933502, 106.2757721
31: -58.6920204, 39.5607071, -58.5457458, 39.5074005, -98.1994171, 98.1064529
32: -51.9725533, 39.5191650, -51.8439980, 39.4226913, -91.3952332, 91.3631592
33: -77.6025543, 50.1773338, -77.4602890, 49.9996681, -127.6022186, 127.6376190
34: -63.5124817, 42.3969765, -63.4088516, 42.2666664, -105.7791214, 105.8058243
35: -59.8537369, 42.2236595, -59.7494698, 42.0830765, -101.9368134, 101.9731293
36: -58.6376724, 42.9619293, -58.5602684, 42.8984985, -101.5361633, 101.5221939
37: -88.3984604, 50.2882462, -88.2804184, 50.1975021, -138.5959625, 138.5686646
38: -71.1473846, 52.7226639, -71.0132446, 52.6205788, -123.7679596, 123.7359085
39: -81.5007553, 44.2839508, -81.3950806, 44.1876411, -125.6884003, 125.6790314
40: -73.4175720, 39.6714554, -73.2916107, 39.5844612, -113.0020294, 112.9630661
41: -54.8887405, 44.0590744, -54.7958221, 43.9882889, -98.8770142, 98.8548965
42: -46.3234444, 39.7741241, -46.2065353, 39.6388474, -85.9622803, 85.9806595

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2651541, upper bound: 78.5359335
time: 72.64 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3798320, upper bound: 78.5359335
time: 67.55 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -74.8477859, 52.7807274, -75.0086746, 52.9050751, -127.7528534, 127.7893982
1: -44.1978912, 40.3060837, -44.2941933, 40.3788147, -84.5767059, 84.6002808
2: -41.2605553, 41.1905479, -41.4146461, 41.3173790, -82.5779266, 82.6051941
3: -49.9488716, 45.7779961, -50.1073151, 45.9146309, -95.8634949, 95.8852997
4: -51.8316917, 48.5682335, -52.0286942, 48.6851807, -100.5168686, 100.5969238
5: -51.0552673, 49.7338905, -51.2154694, 49.9193230, -100.9745941, 100.9493561
6: -57.4114265, 41.5519333, -57.5340080, 41.6952438, -99.1066666, 99.0859299
7: -56.3350830, 45.4200363, -56.4936523, 45.5127945, -101.8478775, 101.9136887
8: -57.2012672, 53.7841415, -57.3482895, 53.9168472, -111.1181030, 111.1324234
9: -51.2375984, 47.0712738, -51.3231125, 47.2576065, -98.4951935, 98.3943863
10: -67.2479019, 59.4227333, -67.4399948, 59.7268753, -126.9747772, 126.8627243
11: -65.2243042, 46.0167389, -65.4046249, 46.2231178, -111.4474182, 111.4213638
12: -60.4402122, 60.6216278, -60.7561226, 60.9876099, -121.4278107, 121.3777466
13: -63.9857521, 57.0136757, -64.1216278, 57.1791229, -121.1648712, 121.1352997
14: -98.7616196, 48.2418365, -99.0440903, 48.5193481, -147.2809753, 147.2859192
15: -60.2237854, 43.2893105, -60.4281769, 43.4345322, -103.6583176, 103.7174835
16: -68.8062744, 46.6991081, -68.9619598, 46.8844986, -115.6907730, 115.6610718
17: -99.9457779, 53.5834961, -100.2292633, 53.8551064, -153.8008728, 153.8127594
18: -58.0206261, 49.1960030, -58.1996651, 49.2914810, -107.3121033, 107.3956451
19: -47.7984657, 34.8986626, -47.9233742, 34.9519234, -82.7503891, 82.8220367
20: -42.9328270, 36.6042709, -43.0541725, 36.7047958, -79.6376190, 79.6584473
21: -59.1064034, 40.2060089, -59.2673912, 40.3278198, -99.4342194, 99.4733887
22: -59.0635147, 40.2739143, -59.2165146, 40.3955460, -99.4590607, 99.4904327
23: -48.8033524, 40.0638123, -48.9132957, 40.1286163, -88.9319687, 88.9771118
24: -53.4612579, 39.4380417, -53.6372948, 39.5299950, -92.9912491, 93.0753326
25: -50.8805084, 45.3702164, -51.0045433, 45.4499435, -96.3304443, 96.3747559
26: -69.5416718, 61.0309410, -69.7620697, 61.2336159, -130.7752838, 130.7930145
27: -58.4461136, 40.6590424, -58.6370430, 40.7365570, -99.1826706, 99.2960815
28: -49.6909218, 44.9596786, -49.7921638, 45.0049744, -94.6958923, 94.7518463
29: -60.4466591, 35.0110550, -60.5597076, 35.1449127, -95.5915680, 95.5707626
30: -61.4555397, 44.7462120, -61.5618134, 44.8641434, -106.3196793, 106.3080292
31: -58.4921684, 39.5288353, -58.6723747, 39.5765800, -98.0687408, 98.2012100
32: -51.8068390, 39.3334198, -51.9712601, 39.4956970, -91.3025208, 91.3046722
33: -77.3361511, 50.0322037, -77.5758820, 50.1799774, -127.5161285, 127.6080856
34: -63.3911057, 42.2848740, -63.5226173, 42.4035835, -105.7946930, 105.8074951
35: -59.6955376, 42.1298332, -59.8575478, 42.2278900, -101.9234238, 101.9873810
36: -58.5416527, 42.8983803, -58.6506271, 42.9612694, -101.5029144, 101.5490112
37: -88.1894760, 50.2186012, -88.3805466, 50.2974472, -138.4869232, 138.5991516
38: -70.9888916, 52.6441727, -71.1684265, 52.7229576, -123.7118530, 123.8125839
39: -81.3104248, 44.2124519, -81.4996872, 44.2915802, -125.6019897, 125.7121429
40: -73.2321014, 39.5942841, -73.3962860, 39.6718407, -112.9039459, 112.9905701
41: -54.7675934, 43.9469681, -54.8865166, 44.0360069, -98.8035889, 98.8334732
42: -46.1958199, 39.5864868, -46.3242531, 39.7717896, -85.9676056, 85.9107285

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3969550
time: 70.68 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3969550
time: 69.08 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -75.1144867, 52.9362373, -75.1249847, 52.9416161, -128.0560913, 128.0612183
1: -44.3556442, 40.3972626, -44.3632774, 40.4007874, -84.7564240, 84.7605438
2: -41.5358429, 41.3353577, -41.5455742, 41.3399620, -82.8758087, 82.8809280
3: -50.2250175, 45.9484596, -50.2387314, 45.9542274, -96.1792450, 96.1871948
4: -52.1782036, 48.7113647, -52.1940765, 48.7170868, -100.8952789, 100.9054337
5: -51.3365974, 49.9555283, -51.3498535, 49.9614639, -101.2980652, 101.3053818
6: -57.5793266, 41.7896996, -57.5864830, 41.8019371, -99.3812637, 99.3761826
7: -56.5897179, 45.5377197, -56.6032982, 45.5445023, -102.1342163, 102.1410141
8: -57.4513359, 53.9467888, -57.4621391, 53.9525299, -111.4038696, 111.4089279
9: -51.3597565, 47.3795395, -51.3643951, 47.3949432, -98.7546997, 98.7439194
10: -67.4915009, 59.9411163, -67.4996796, 59.9656219, -127.4571075, 127.4407959
11: -65.4511642, 46.3736534, -65.4572906, 46.3927956, -111.8439636, 111.8309402
12: -60.7931366, 61.2815475, -60.8008080, 61.3071632, -122.1002960, 122.0823517
13: -64.1553345, 57.2698059, -64.1637115, 57.2864189, -121.4417419, 121.4335175
14: -99.1130600, 48.7392731, -99.1253738, 48.7609558, -147.8739929, 147.8646545
15: -60.5680923, 43.4740753, -60.5883026, 43.4787827, -104.0468674, 104.0623779
16: -69.0204926, 46.9978600, -69.0277252, 47.0180016, -116.0384903, 116.0255890
17: -100.2750931, 54.0699120, -100.2845993, 54.0876007, -154.3627014, 154.3545074
18: -58.2881775, 49.3375626, -58.3066330, 49.3461418, -107.6343079, 107.6441956
19: -47.9614944, 34.9811554, -47.9741211, 34.9871025, -82.9485931, 82.9552689
20: -43.0959549, 36.7726212, -43.1006432, 36.7812767, -79.8772278, 79.8732605
21: -59.3130493, 40.4112244, -59.3196373, 40.4206848, -99.7337265, 99.7308578
22: -59.2643051, 40.4637146, -59.2922020, 40.4705048, -99.7348099, 99.7559204
23: -48.9534492, 40.1572418, -48.9611740, 40.1654472, -89.1188965, 89.1184158
24: -53.7491837, 39.5476379, -53.7653542, 39.5515060, -93.3006744, 93.3129883
25: -51.0640335, 45.4888420, -51.0764771, 45.4926109, -96.5566406, 96.5653152
26: -69.8118591, 61.3883858, -69.8217087, 61.4018288, -131.2136841, 131.2100983
27: -58.7565956, 40.7522736, -58.7735939, 40.7583008, -99.5148926, 99.5258636
28: -49.8297310, 45.0277138, -49.8361626, 45.0330925, -94.8628235, 94.8638763
29: -60.5986633, 35.2449875, -60.6103745, 35.2534790, -95.8521423, 95.8553619
30: -61.6026764, 44.9328651, -61.6097336, 44.9454079, -106.5480804, 106.5426025
31: -58.7513542, 39.5964508, -58.7691803, 39.6020775, -98.3534317, 98.3656311
32: -52.0139313, 39.6098366, -52.0204697, 39.6264877, -91.6404114, 91.6303024
33: -77.7243195, 50.2160225, -77.7433472, 50.2216454, -127.9459686, 127.9593658
34: -63.5912781, 42.4318657, -63.6042595, 42.4377823, -106.0290604, 106.0361252
35: -59.9498253, 42.2536964, -59.9654236, 42.2578049, -102.2076187, 102.2191162
36: -58.6853981, 42.9892883, -58.6950760, 42.9959259, -101.6813126, 101.6843643
37: -88.4630127, 50.3350449, -88.4822540, 50.3408775, -138.8038940, 138.8172913
38: -71.2324829, 52.7512169, -71.2460861, 52.7575493, -123.9900360, 123.9972916
39: -81.5671463, 44.3140984, -81.5902557, 44.3197250, -125.8868713, 125.9043427
40: -73.4848557, 39.6898613, -73.4967804, 39.6941605, -113.1790161, 113.1866379
41: -54.9433899, 44.0923653, -54.9530983, 44.0964584, -99.0398483, 99.0454636
42: -46.3602829, 39.8886604, -46.3661499, 39.9118614, -86.2721405, 86.2548065

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2646196, upper bound: 78.5367813
time: 75.05 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3969550
time: 347.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 424.58 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.2646196, upper bound: 78.2444293
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.2646196, upper bound: 78.2444898
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3897204
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.3798320, upper bound: 78.3897204
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.4254198, upper bound: 78.2492643
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.2646196, upper bound: 78.2492643
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3914811
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3914811
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3946298
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3946298
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.2651541, upper bound: 78.5359335
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.3798320, upper bound: 78.5359335
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3969550
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3969550
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.2646196, upper bound: 78.5367813
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 424.58
Output dim: 12, lower bound: -78.2646196, upper bound: 78.3969550

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -74.3068542, 52.5266190, -74.2683182, 52.4275589, -126.7344055, 126.7949219
1: -43.8749542, 40.1497498, -43.8339043, 40.0685806, -83.9435196, 83.9836578
2: -40.7179947, 40.9935074, -40.6936531, 40.9305992, -81.6485901, 81.6871567
3: -49.3977585, 45.5144653, -49.3424492, 45.4022522, -94.8000031, 94.8569183
4: -51.1124382, 48.3301048, -51.0915413, 48.2049942, -99.3174286, 99.4216461
5: -50.5093880, 49.4327774, -50.4647446, 49.3429413, -99.8523254, 99.8975220
6: -57.0940208, 41.1746483, -57.0859375, 41.1781425, -98.2721634, 98.2605820
7: -55.8829498, 45.2095604, -55.8705444, 45.1536407, -101.0365906, 101.0801086
8: -56.6886215, 53.5241966, -56.6688690, 53.4477463, -110.1363678, 110.1930695
9: -50.9157333, 46.4781303, -50.9349136, 46.5328751, -97.4485855, 97.4130402
10: -66.6722870, 58.3100357, -66.7277374, 58.4280701, -125.1003571, 125.0377731
11: -64.7654572, 45.1865997, -64.6887970, 45.1701088, -109.9355469, 109.8753967
12: -59.9201889, 59.3964119, -59.8753471, 59.4693298, -119.3895111, 119.2717590
13: -63.6884689, 56.5266647, -63.7719307, 56.5860405, -120.2745056, 120.2985992
14: -98.1289444, 47.1717758, -98.1839371, 47.2556152, -145.3845520, 145.3557129
15: -59.4677086, 42.9642448, -59.4430962, 42.9345093, -102.4022217, 102.4073410
16: -68.3757935, 46.0348701, -68.4217224, 46.0635986, -114.4393921, 114.4565887
17: -99.4335632, 52.6483383, -99.4061737, 52.6979828, -152.1315155, 152.0545044
18: -57.5076599, 48.9230957, -57.5501900, 48.9208755, -106.4285355, 106.4732819
19: -47.5149193, 34.6688766, -47.4521942, 34.5771484, -82.0920715, 82.1210709
20: -42.6331787, 36.2421608, -42.6170464, 36.2012138, -78.8343964, 78.8592072
21: -58.7259865, 39.7346268, -58.6668015, 39.6559525, -98.3819275, 98.4014206
22: -58.6361771, 39.9485016, -58.6941948, 39.9438133, -98.5799866, 98.6427002
23: -48.5527802, 39.8136978, -48.5241318, 39.7543411, -88.3071213, 88.3378296
24: -52.9728851, 39.2530441, -53.0832901, 39.3358345, -92.3087158, 92.3363342
25: -50.5901871, 45.0897217, -50.6122971, 45.0523453, -95.6425323, 95.7020187
26: -69.1176682, 60.3396111, -69.0438995, 60.3086777, -129.4263458, 129.3834991
27: -57.8434525, 40.4479141, -57.9656601, 40.5193405, -98.3627930, 98.4135742
28: -49.4374466, 44.7168465, -49.4162216, 44.6818619, -94.1193085, 94.1330566
29: -60.1488037, 34.5991211, -60.1145172, 34.5649605, -94.7137604, 94.7136383
30: -61.1843185, 44.2902489, -61.1485367, 44.2421608, -105.4264832, 105.4387817
31: -58.0860977, 39.3225021, -58.1270103, 39.2414932, -97.3275757, 97.4495087
32: -51.4747734, 38.8552589, -51.5085106, 38.8898201, -90.3645935, 90.3637695
33: -76.5947418, 49.6619835, -76.6696625, 49.7104340, -126.3051758, 126.3316269
34: -62.9038467, 41.9620895, -62.9014130, 41.9586868, -104.8625336, 104.8634949
35: -59.1536293, 41.8171158, -59.2340393, 41.8668137, -101.0204315, 101.0511551
36: -58.2158241, 42.6917496, -58.2549210, 42.7064323, -100.9222565, 100.9466705
37: -87.6418839, 49.9267082, -87.7170334, 49.9147949, -137.5566711, 137.6437378
38: -70.5472107, 52.3879852, -70.5887222, 52.4189415, -122.9661560, 122.9767075
39: -80.8466721, 44.0210114, -80.8894653, 43.9860077, -124.8326721, 124.9104691
40: -72.7179871, 39.3718185, -72.7859192, 39.3709297, -112.0889053, 112.1577301
41: -54.3852119, 43.7095261, -54.4201622, 43.7023468, -98.0875549, 98.1296844
42: -45.8943481, 39.0470543, -45.8885765, 39.0806503, -84.9749908, 84.9356308

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.1339610
time: 65.43 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2311071
time: 73.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -74.4462280, 52.5698395, -74.5830612, 52.6159821, -127.0622101, 127.1529007
1: -43.9600029, 40.1734161, -44.0341110, 40.2018433, -84.1618500, 84.2075195
2: -40.8430939, 41.0219803, -40.9624939, 41.0983467, -81.9414368, 81.9844742
3: -49.5476227, 45.5573463, -49.6564827, 45.6197853, -95.1674042, 95.2138290
4: -51.2965202, 48.3624077, -51.4720764, 48.4286041, -99.7251205, 99.8344803
5: -50.6463699, 49.4833717, -50.7517128, 49.5960808, -100.2424393, 100.2350845
6: -57.1463318, 41.2996750, -57.2886543, 41.4545708, -98.6008987, 98.5883255
7: -55.9726028, 45.2488136, -56.0961800, 45.2856140, -101.2582169, 101.3449783
8: -56.8071404, 53.5659904, -56.9310875, 53.6608772, -110.4680023, 110.4970703
9: -50.9817085, 46.5882034, -51.1013412, 46.7969589, -97.7786713, 97.6895447
10: -66.7390747, 58.4908485, -66.9768372, 58.8393173, -125.5783920, 125.4676819
11: -64.8284302, 45.4138298, -65.0547180, 45.6372719, -110.4656982, 110.4685516
12: -59.9797173, 59.6822395, -60.2695999, 60.0619125, -120.0416260, 119.9518433
13: -63.7392960, 56.6261063, -63.9505463, 56.8273315, -120.5666046, 120.5766373
14: -98.2128143, 47.4385300, -98.5447464, 47.8021660, -146.0149689, 145.9832764
15: -59.6889076, 43.0083160, -59.9070816, 43.1938057, -102.8827133, 102.9153976
16: -68.4417343, 46.1841202, -68.6555786, 46.3970871, -114.8388214, 114.8396988
17: -99.5066147, 52.9166222, -99.8440933, 53.2549477, -152.7615662, 152.7607117
18: -57.6104012, 48.9809647, -57.8034668, 49.0783234, -106.6887207, 106.7844315
19: -47.5663490, 34.7737160, -47.6882668, 34.7984810, -82.3648224, 82.4619827
20: -42.6792641, 36.3657990, -42.8084679, 36.4639969, -79.1432648, 79.1742630
21: -58.7808609, 39.8896484, -58.9522362, 39.9870148, -98.7678680, 98.8418884
22: -58.7198143, 40.0152664, -58.9229813, 40.1281281, -98.8479462, 98.9382477
23: -48.5990562, 39.9008408, -48.7093201, 39.9472580, -88.5463104, 88.6101456
24: -53.0675163, 39.2782593, -53.3067665, 39.4332657, -92.5007782, 92.5850220
25: -50.6552048, 45.1695938, -50.7989388, 45.2458076, -95.9010162, 95.9685211
26: -69.1856384, 60.4960594, -69.3161392, 60.6486053, -129.8342438, 129.8121948
27: -57.9453888, 40.4772987, -58.2057953, 40.6144409, -98.5598297, 98.6830902
28: -49.4787369, 44.8059654, -49.5964088, 44.8699417, -94.3486786, 94.4023743
29: -60.1996002, 34.7235489, -60.3284607, 34.8361893, -95.0357895, 95.0520096
30: -61.2333641, 44.4188385, -61.3209763, 44.5234146, -105.7567749, 105.7398148
31: -58.1526146, 39.4142227, -58.3409195, 39.4409561, -97.5935516, 97.7551422
32: -51.5229645, 38.9777908, -51.7171059, 39.1497765, -90.6727219, 90.6949005
33: -76.7609100, 49.7070618, -77.0645599, 49.9051056, -126.6660080, 126.7716141
34: -63.0323753, 42.0039673, -63.1896591, 42.1819687, -105.2143402, 105.1936264
35: -59.2478409, 41.8495026, -59.4621506, 42.0089951, -101.2568359, 101.3116531
36: -58.2632408, 42.7356224, -58.4153976, 42.8248444, -101.0880890, 101.1510162
37: -87.7425766, 49.9957619, -88.0028076, 50.0925484, -137.8351288, 137.9985657
38: -70.6199493, 52.4250336, -70.8156281, 52.5390511, -123.1589966, 123.2406616
39: -80.9447861, 44.0495567, -81.1731720, 44.1189041, -125.0636902, 125.2227249
40: -72.8137589, 39.4095039, -73.0367432, 39.5280724, -112.3418274, 112.4462433
41: -54.4561310, 43.7760391, -54.6167183, 43.8649750, -98.3211060, 98.3927383
42: -45.9461594, 39.1697884, -46.0975838, 39.3442650, -85.2904205, 85.2673721

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.1361027
time: 79.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2311685
time: 74.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -74.5657883, 52.6837959, -74.3811569, 52.4640503, -127.0298157, 127.0649490
1: -44.0231743, 40.2416573, -43.8972282, 40.0904007, -84.1135712, 84.1388855
2: -40.9864883, 41.1382446, -40.8215485, 40.9527664, -81.9392548, 81.9597931
3: -49.6702194, 45.6837006, -49.4726334, 45.4408684, -95.1110687, 95.1563263
4: -51.4463806, 48.4713974, -51.2516518, 48.2367020, -99.6830750, 99.7230530
5: -50.7874680, 49.6520844, -50.5983124, 49.3843460, -100.1718140, 100.2503891
6: -57.2548904, 41.3937263, -57.1361885, 41.2783966, -98.5332794, 98.5299149
7: -56.1292839, 45.3266487, -55.9753304, 45.1849518, -101.3142395, 101.3019714
8: -56.9269524, 53.6855011, -56.7772560, 53.4825478, -110.4095001, 110.4627533
9: -51.0371552, 46.7783623, -50.9758377, 46.6673584, -97.7045135, 97.7541962
10: -66.9128265, 58.8164864, -66.7866211, 58.6625938, -125.5754166, 125.6031036
11: -64.9892654, 45.5348511, -64.7399979, 45.3370285, -110.3262939, 110.2748489
12: -60.2730484, 60.0477562, -59.9193268, 59.7857132, -120.0587616, 119.9670868
13: -63.8554535, 56.7667847, -63.8121910, 56.6847496, -120.5401840, 120.5789795
14: -98.4778137, 47.6648712, -98.2637253, 47.4943733, -145.9721680, 145.9285889
15: -59.8018646, 43.1384850, -59.5985374, 42.9755096, -102.7773743, 102.7369995
16: -68.5782013, 46.3232079, -68.4826279, 46.1922264, -114.7704163, 114.8058319
17: -99.7599106, 53.1284409, -99.4602127, 52.9270477, -152.6869507, 152.5886383
18: -57.7569885, 49.0606537, -57.6449013, 48.9737854, -106.7307739, 106.7055511
19: -47.6740417, 34.7480202, -47.5024223, 34.6107864, -82.2848282, 82.2504349
20: -42.7953720, 36.4066544, -42.6626892, 36.2761536, -79.0715179, 79.0693359
21: -58.9302902, 39.9366798, -58.7174149, 39.7471008, -98.6773911, 98.6540985
22: -58.8283768, 40.1266251, -58.7664146, 40.0129471, -98.8413239, 98.8930359
23: -48.6997108, 39.9032669, -48.5712547, 39.7894897, -88.4891891, 88.4745178
24: -53.2487183, 39.3609695, -53.2061729, 39.3570366, -92.6057434, 92.5671310
25: -50.7683372, 45.2017517, -50.6824036, 45.0911751, -95.8595123, 95.8841553
26: -69.3834000, 60.6964645, -69.1022797, 60.4754868, -129.8588867, 129.7987366
27: -58.1389198, 40.5357056, -58.0940628, 40.5400734, -98.6789856, 98.6297684
28: -49.5738754, 44.7805786, -49.4592094, 44.7080956, -94.2819672, 94.2397842
29: -60.2968292, 34.8260269, -60.1638947, 34.6692238, -94.9660492, 94.9899216
30: -61.3265572, 44.4669037, -61.1953850, 44.3185654, -105.6451263, 105.6622849
31: -58.3314171, 39.3870392, -58.2186203, 39.2655563, -97.5969696, 97.6056519
32: -51.6788025, 39.1238747, -51.5569000, 39.0186119, -90.6974030, 90.6807709
33: -76.9757385, 49.8447227, -76.8343353, 49.7516479, -126.7273788, 126.6790466
34: -63.0985641, 42.1063614, -62.9806213, 41.9917297, -105.0902939, 105.0869827
35: -59.4004822, 41.9393616, -59.3390503, 41.8962631, -101.2967377, 101.2784042
36: -58.3567200, 42.7763977, -58.2980995, 42.7388535, -101.0955734, 101.0744934
37: -87.9000168, 50.0391159, -87.8110962, 49.9567909, -137.8567810, 137.8502197
38: -70.7843246, 52.4928818, -70.6633759, 52.4533424, -123.2376633, 123.1562500
39: -81.0948944, 44.1208725, -80.9761810, 44.0137291, -125.1086121, 125.0970535
40: -72.9585114, 39.4654617, -72.8777313, 39.3927498, -112.3512573, 112.3431931
41: -54.5526123, 43.8491287, -54.4824371, 43.7605972, -98.3132095, 98.3315659
42: -46.0569687, 39.3303719, -45.9311371, 39.2125778, -85.2695465, 85.2615051

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2320601
time: 78.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3814815
time: 65.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -74.7083206, 52.7270737, -74.6983490, 52.6526108, -127.3609314, 127.4254227
1: -44.1116180, 40.2655411, -44.0991898, 40.2237167, -84.3353348, 84.3647308
2: -41.1127548, 41.1672173, -41.0915565, 41.1207809, -82.2335281, 82.2587738
3: -49.8212090, 45.7276573, -49.7875938, 45.6590958, -95.4803009, 95.5152435
4: -51.6334000, 48.5041046, -51.6347733, 48.4603233, -100.0937195, 100.1388779
5: -50.9257774, 49.7033463, -50.8864250, 49.6380882, -100.5638580, 100.5897675
6: -57.3090630, 41.5210800, -57.3401337, 41.5568123, -98.8658752, 98.8612137
7: -56.2219658, 45.3667641, -56.2033653, 45.3174057, -101.5393677, 101.5701141
8: -57.0485611, 53.7279625, -57.0420341, 53.6961479, -110.7446899, 110.7699890
9: -51.1042938, 46.8897095, -51.1427917, 46.9326477, -98.0369263, 98.0325012
10: -66.9808655, 58.9992485, -67.0358200, 59.0757065, -126.0565720, 126.0350647
11: -65.0540314, 45.7637405, -65.1068573, 45.8056641, -110.8596954, 110.8705978
12: -60.3332481, 60.3352165, -60.3138428, 60.3799553, -120.7132034, 120.6490479
13: -63.9075127, 56.8724442, -63.9918518, 56.9298706, -120.8373871, 120.8642883
14: -98.5633087, 47.9324303, -98.6254501, 48.0392303, -146.6025391, 146.5578766
15: -60.0259018, 43.1863174, -60.0647202, 43.2365189, -103.2624207, 103.2510376
16: -68.6485138, 46.4745979, -68.7194672, 46.5277328, -115.1762314, 115.1940613
17: -99.8340912, 53.3983498, -99.8988800, 53.4855118, -153.3195801, 153.2972260
18: -57.8670845, 49.1196785, -57.9054794, 49.1320610, -106.9991455, 107.0251465
19: -47.7269897, 34.8536072, -47.7391624, 34.8328018, -82.5597916, 82.5927582
20: -42.8423805, 36.5312920, -42.8545265, 36.5397720, -79.3821564, 79.3858185
21: -58.9862556, 40.0926094, -59.0036736, 40.0789032, -99.0651550, 99.0962830
22: -58.9136086, 40.1978455, -58.9967041, 40.2003098, -99.1139069, 99.1945419
23: -48.7471886, 39.9912491, -48.7567596, 39.9831200, -88.7303009, 88.7480087
24: -53.3464584, 39.3866577, -53.4321404, 39.4545441, -92.8009949, 92.8187866
25: -50.8350105, 45.2844887, -50.8703613, 45.2870293, -96.1220398, 96.1548462
26: -69.4521942, 60.8539734, -69.3754044, 60.8164253, -130.2686005, 130.2293701
27: -58.2460632, 40.5658302, -58.3381729, 40.6355743, -98.8816299, 98.9040070
28: -49.6159935, 44.8709412, -49.6394310, 44.8972168, -94.5132141, 94.5103760
29: -60.3489609, 34.9557152, -60.3783798, 34.9444733, -95.2934341, 95.3340912
30: -61.3773193, 44.6005249, -61.3689461, 44.6036682, -105.9809875, 105.9694672
31: -58.4026413, 39.4799500, -58.4343719, 39.4657173, -97.8683548, 97.9143219
32: -51.7278671, 39.2480774, -51.7661362, 39.2799377, -91.0078049, 91.0142136
33: -77.1435471, 49.8904800, -77.2306519, 49.9466209, -127.0901642, 127.1211319
34: -63.2283363, 42.1492004, -63.2698364, 42.2156219, -105.4439545, 105.4190369
35: -59.4964409, 41.9721527, -59.5686111, 42.0387955, -101.5352325, 101.5407639
36: -58.4049530, 42.8221397, -58.4591103, 42.8582382, -101.2631912, 101.2812500
37: -88.0051575, 50.1094589, -88.1004333, 50.1356430, -138.1407928, 138.2098999
38: -70.8579865, 52.5307579, -70.8910828, 52.5735245, -123.4315109, 123.4218445
39: -81.1954880, 44.1501083, -81.2621536, 44.1469078, -125.3423767, 125.4122543
40: -73.0608749, 39.5035477, -73.1338806, 39.5501137, -112.6109848, 112.6374283
41: -54.6264534, 43.9173775, -54.6809273, 43.9247437, -98.5511932, 98.5983047
42: -46.1084824, 39.4552917, -46.1397057, 39.4781075, -85.5865936, 85.5949936

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2325801
time: 65.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3814815
time: 65.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -74.4255219, 52.5527458, -74.5532227, 52.6596451, -127.0851440, 127.1059647
1: -43.9537163, 40.1664543, -44.0200768, 40.2113342, -84.1650543, 84.1865234
2: -40.8570175, 41.0113754, -41.0010796, 41.1139946, -81.9710083, 82.0124512
3: -49.5405960, 45.5425568, -49.6585960, 45.6390076, -95.1796036, 95.2011490
4: -51.2707253, 48.3536186, -51.4443092, 48.4093475, -99.6800690, 99.7979279
5: -50.6571045, 49.4609261, -50.7957649, 49.6060371, -100.2631378, 100.2566833
6: -57.1403160, 41.2173424, -57.2477837, 41.3075943, -98.4479065, 98.4651260
7: -55.9968185, 45.2310104, -56.1561279, 45.3272018, -101.3240204, 101.3871384
8: -56.8125954, 53.5506439, -56.9519081, 53.6446457, -110.4572296, 110.5025482
9: -50.9528923, 46.5932693, -51.0987968, 46.8034592, -97.7563477, 97.6920547
10: -66.7300797, 58.5526161, -67.1024857, 58.9563026, -125.6863861, 125.6550980
11: -64.8041992, 45.3464966, -64.9674225, 45.5140190, -110.3182220, 110.3139114
12: -59.9511452, 59.6606979, -60.2954597, 60.0406380, -119.9917755, 119.9561539
13: -63.7243309, 56.5921173, -63.8838425, 56.7805939, -120.5049057, 120.4759445
14: -98.1973038, 47.3526001, -98.5758286, 47.6482620, -145.8455658, 145.9284363
15: -59.6005440, 43.0026855, -59.7518158, 43.1080894, -102.7086334, 102.7544937
16: -68.4292297, 46.1487350, -68.6374512, 46.3376236, -114.7668533, 114.7861862
17: -99.4751129, 52.7868080, -99.7229614, 53.0266113, -152.5017242, 152.5097656
18: -57.5709267, 48.9803429, -57.7919426, 49.0673332, -106.6382599, 106.7722855
19: -47.5528564, 34.7158127, -47.6199951, 34.6873589, -82.2402191, 82.3358002
20: -42.6720581, 36.3065491, -42.8009872, 36.3459930, -79.0180435, 79.1075363
21: -58.7645187, 39.8349342, -58.9081879, 39.8835068, -98.6480103, 98.7431107
22: -58.6797523, 40.0152359, -58.8505020, 40.1279831, -98.8077316, 98.8657379
23: -48.5854111, 39.8612518, -48.6669235, 39.8774872, -88.4628983, 88.5281677
24: -53.0296249, 39.2722435, -53.2412224, 39.4018631, -92.4314880, 92.5134659
25: -50.6221237, 45.1438026, -50.7279320, 45.1972275, -95.8193436, 95.8717346
26: -69.1616516, 60.5144272, -69.4018555, 60.7060242, -129.8676758, 129.9162903
27: -57.9255600, 40.4694977, -58.1814613, 40.6053696, -98.5309296, 98.6509552
28: -49.4703026, 44.7448196, -49.5388603, 44.7652740, -94.2355728, 94.2836761
29: -60.1822815, 34.6879082, -60.2586174, 34.7745895, -94.9568710, 94.9465256
30: -61.2163773, 44.3849297, -61.3186646, 44.4620056, -105.6783676, 105.7035980
31: -58.1425552, 39.3576355, -58.3426590, 39.3338928, -97.4764481, 97.7002869
32: -51.5157394, 38.9444160, -51.6840401, 39.0905991, -90.6063232, 90.6284485
33: -76.7147522, 49.7006760, -76.9491119, 49.9318123, -126.6465607, 126.6497879
34: -62.9813461, 41.9965591, -63.0936699, 42.1284752, -105.1098175, 105.0902252
35: -59.2483025, 41.8477287, -59.4466438, 42.0410995, -101.2893982, 101.2943573
36: -58.2620010, 42.7189026, -58.3869781, 42.7989311, -101.0609283, 101.1058807
37: -87.7013092, 49.9738846, -87.9117050, 50.0562515, -137.7575684, 137.8855896
38: -70.6305389, 52.4165154, -70.8169022, 52.5550766, -123.1856079, 123.2334061
39: -80.9108658, 44.0513763, -81.0802460, 44.1175919, -125.0284500, 125.1316223
40: -72.7807312, 39.3896141, -72.9826965, 39.4789925, -112.2597198, 112.3723145
41: -54.4375458, 43.7439079, -54.5729675, 43.8096390, -98.2471695, 98.3168716
42: -45.9310188, 39.1500664, -46.0476837, 39.3434143, -85.2744293, 85.1977539

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.1437883
time: 78.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2365052
time: 80.26 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -74.5648651, 52.5958328, -74.8702240, 52.8482819, -127.4131470, 127.4660492
1: -44.0385704, 40.1900330, -44.2223129, 40.3448029, -84.3833771, 84.4123459
2: -40.9819946, 41.0398140, -41.2708511, 41.2821732, -82.2641602, 82.3106689
3: -49.6903763, 45.5853386, -49.9742737, 45.8569145, -95.5472870, 95.5596008
4: -51.4546928, 48.3858643, -51.8260841, 48.6335793, -100.0882721, 100.2119446
5: -50.7940025, 49.5114365, -51.0841026, 49.8595619, -100.6535645, 100.5955353
6: -57.1926498, 41.3422394, -57.4519119, 41.5850029, -98.7776337, 98.7941284
7: -56.0867119, 45.2702103, -56.3854790, 45.4594803, -101.5461807, 101.6556854
8: -56.9309273, 53.5923042, -57.2155075, 53.8585167, -110.7894440, 110.8078079
9: -51.0187607, 46.7032738, -51.2657814, 47.0692940, -98.0880585, 97.9690552
10: -66.7967987, 58.7330933, -67.3516388, 59.3688736, -126.1656494, 126.0847321
11: -64.8671417, 45.5736275, -65.3340302, 45.9821854, -110.8493195, 110.9076538
12: -60.0105591, 59.9465065, -60.6904259, 60.6343803, -120.6449356, 120.6369247
13: -63.7751389, 56.6916008, -64.0636139, 57.0278702, -120.8030014, 120.7552032
14: -98.2810211, 47.6193657, -98.9377213, 48.1961555, -146.4771729, 146.5570679
15: -59.8216629, 43.0466042, -60.2174950, 43.3710899, -103.1927490, 103.2640915
16: -68.4949951, 46.2986832, -68.8773499, 46.6728630, -115.1678543, 115.1760330
17: -99.5480652, 53.0551949, -100.1616592, 53.5852203, -153.1332703, 153.2168579
18: -57.6750374, 49.0381355, -58.0505981, 49.2258224, -106.9008636, 107.0887299
19: -47.6041641, 34.8205414, -47.8562126, 34.9094696, -82.5136261, 82.6767426
20: -42.7180634, 36.4301643, -42.9924736, 36.6096649, -79.3277283, 79.4226379
21: -58.8192978, 39.9898224, -59.1944466, 40.2151642, -99.0344391, 99.1842651
22: -58.7634239, 40.0821991, -59.0802612, 40.3157959, -99.0792236, 99.1624603
23: -48.6315575, 39.9484253, -48.8526993, 40.0711784, -88.7027283, 88.8011169
24: -53.1247177, 39.2974091, -53.4683609, 39.4999161, -92.6246338, 92.7657700
25: -50.6871223, 45.2237968, -50.9150391, 45.3926239, -96.0797424, 96.1388245
26: -69.2294769, 60.6708221, -69.6744995, 61.0483322, -130.2778015, 130.3453217
27: -58.0276337, 40.4987869, -58.4259834, 40.7014160, -98.7290497, 98.9247742
28: -49.5115089, 44.8338470, -49.7195930, 44.9545708, -94.4660797, 94.5534363
29: -60.2329826, 34.8123970, -60.4732666, 35.0483017, -95.2812805, 95.2856598
30: -61.2652817, 44.5138283, -61.4920082, 44.7440834, -106.0093613, 106.0058212
31: -58.2091370, 39.4492683, -58.5615044, 39.5341835, -97.7433167, 98.0107727
32: -51.5638313, 39.0668068, -51.8933525, 39.3516464, -90.9154816, 90.9601517
33: -76.8810425, 49.7456856, -77.3456879, 50.1267509, -127.0077744, 127.0913696
34: -63.1099701, 42.0383072, -63.3835144, 42.3522530, -105.4622192, 105.4218140
35: -59.3425140, 41.8799667, -59.6762733, 42.1839447, -101.5264587, 101.5562439
36: -58.3094406, 42.7626572, -58.5483627, 42.9210243, -101.2304535, 101.3110199
37: -87.8020325, 50.0427971, -88.1998520, 50.2357178, -138.0377502, 138.2426453
38: -70.7034531, 52.4534416, -71.0452728, 52.6758957, -123.3793411, 123.4987183
39: -81.0088196, 44.0798721, -81.3654861, 44.2509384, -125.2597580, 125.4453583
40: -72.8767624, 39.4272461, -73.2368698, 39.6368866, -112.5136414, 112.6641083
41: -54.5083733, 43.8103485, -54.7711029, 43.9729424, -98.4813156, 98.5814514
42: -45.9827309, 39.2733345, -46.2572021, 39.6090889, -85.5918198, 85.5305328

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.1453344
time: 74.78 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.5222625, upper bound: 78.2365164
time: 64.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -74.6845551, 52.7096481, -74.6662292, 52.6958618, -127.3804169, 127.3758774
1: -44.1020508, 40.2580872, -44.0836334, 40.2328911, -84.3349457, 84.3417206
2: -41.1253891, 41.1560440, -41.1290092, 41.1360817, -82.2614670, 82.2850494
3: -49.8130836, 45.7116241, -49.7889748, 45.6772079, -95.4902954, 95.5005951
4: -51.6046867, 48.4948807, -51.6044998, 48.4409637, -100.0456543, 100.0993805
5: -50.9352417, 49.6799316, -50.9295731, 49.6470985, -100.5823364, 100.6095047
6: -57.3012466, 41.4383621, -57.2977600, 41.4099236, -98.7111664, 98.7361221
7: -56.2413826, 45.3480301, -56.2586327, 45.3581238, -101.5995026, 101.6066589
8: -57.0509834, 53.7117310, -57.0605278, 53.6792450, -110.7302246, 110.7722473
9: -51.0737915, 46.8936348, -51.1391830, 46.9380951, -98.0118790, 98.0328217
10: -66.9702377, 59.0588455, -67.1607971, 59.1908722, -126.1611099, 126.2196350
11: -65.0278473, 45.6945305, -65.0182343, 45.6808472, -110.7086945, 110.7127686
12: -60.3038483, 60.3120651, -60.3391418, 60.3572006, -120.6610413, 120.6511993
13: -63.8911400, 56.8328133, -63.9237328, 56.8795357, -120.7706757, 120.7565384
14: -98.5459518, 47.8458405, -98.6550598, 47.8870010, -146.4329224, 146.5009003
15: -59.9343033, 43.1770897, -59.9070702, 43.1487732, -103.0830688, 103.0841599
16: -68.6317596, 46.4375229, -68.6980133, 46.4668732, -115.0986328, 115.1355362
17: -99.8013000, 53.2671661, -99.7768326, 53.2557259, -153.0570221, 153.0440063
18: -57.8220558, 49.1177673, -57.8873444, 49.1201553, -106.9422073, 107.0051117
19: -47.7120018, 34.7948570, -47.6697769, 34.7212143, -82.4332123, 82.4646301
20: -42.8339081, 36.4710083, -42.8459091, 36.4209137, -79.2548218, 79.3169174
21: -58.9686241, 40.0368271, -58.9586067, 39.9746475, -98.9432526, 98.9954300
22: -58.8718414, 40.1941147, -58.9224930, 40.1973114, -99.0691452, 99.1166000
23: -48.7324867, 39.9508209, -48.7139893, 39.9128189, -88.6453094, 88.6648102
24: -53.3066750, 39.3799591, -53.3651428, 39.4228210, -92.7294922, 92.7451019
25: -50.8005829, 45.2558784, -50.7978020, 45.2360878, -96.0366669, 96.0536728
26: -69.4268188, 60.8711624, -69.4594879, 60.8731117, -130.2999268, 130.3306580
27: -58.2216415, 40.5571899, -58.3101273, 40.6259308, -98.8475647, 98.8673172
28: -49.6066475, 44.8084450, -49.5819244, 44.7913971, -94.3980408, 94.3903656
29: -60.3301811, 34.9134827, -60.3074379, 34.8765259, -95.2067108, 95.2209167
30: -61.3587608, 44.5618858, -61.3650551, 44.5383606, -105.8971176, 105.9269409
31: -58.3889465, 39.4220314, -58.4344826, 39.3577423, -97.7466736, 97.8565140
32: -51.7195015, 39.2131653, -51.7318878, 39.2195206, -90.9390106, 90.9450455
33: -77.0959320, 49.8830681, -77.1138382, 49.9725533, -127.0684814, 126.9969025
34: -63.1762924, 42.1406937, -63.1730957, 42.1611290, -105.3374176, 105.3137894
35: -59.4952850, 41.9697227, -59.5517502, 42.0702667, -101.5655518, 101.5214615
36: -58.4028625, 42.8036766, -58.4301262, 42.8312454, -101.2341003, 101.2338028
37: -87.9599152, 50.0862198, -88.0057144, 50.0980759, -138.0579834, 138.0919342
38: -70.8669510, 52.5213089, -70.8920898, 52.5891953, -123.4561386, 123.4133987
39: -81.1591263, 44.1511879, -81.1669998, 44.1452179, -125.3043442, 125.3181915
40: -73.0219727, 39.4831772, -73.0745697, 39.5007935, -112.5227661, 112.5577469
41: -54.6050301, 43.8835411, -54.6352654, 43.8678818, -98.4729156, 98.5188065
42: -46.0934410, 39.4346619, -46.0900040, 39.4766808, -85.5701218, 85.5246582

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2369164
time: 62.63 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3829420
time: 61.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -74.8278122, 52.7530518, -74.9871140, 52.8849182, -127.7127075, 127.7401657
1: -44.1916656, 40.2821274, -44.2898254, 40.3667793, -84.5584335, 84.5719452
2: -41.2517242, 41.1852188, -41.4003525, 41.3048248, -82.5565491, 82.5855637
3: -49.9641495, 45.7559204, -50.1059418, 45.8966484, -95.8607941, 95.8618622
4: -51.7923279, 48.5277252, -51.9902000, 48.6655121, -100.4578400, 100.5179214
5: -51.0736465, 49.7315292, -51.2193832, 49.9019318, -100.9755707, 100.9509125
6: -57.3560944, 41.5658913, -57.5042496, 41.6898193, -99.0458984, 99.0701370
7: -56.3359108, 45.3883591, -56.4930992, 45.4912910, -101.8272018, 101.8814545
8: -57.1732864, 53.7544594, -57.3280907, 53.8942451, -111.0675201, 111.0825348
9: -51.1410866, 47.0051193, -51.3071136, 47.2055397, -98.3466263, 98.3122253
10: -67.0386658, 59.2416496, -67.4110260, 59.6059151, -126.6445770, 126.6526794
11: -65.0931168, 45.9236336, -65.3868942, 46.1509552, -111.2440720, 111.3105316
12: -60.3644104, 60.5997276, -60.7351303, 60.9530182, -121.3174286, 121.3348465
13: -63.9437332, 56.9407768, -64.1054459, 57.1342010, -121.0779266, 121.0462189
14: -98.6318817, 48.1141129, -99.0189209, 48.4340515, -147.0659332, 147.1330261
15: -60.1589241, 43.2260284, -60.3762550, 43.4153900, -103.5743103, 103.6022797
16: -68.7030792, 46.5899467, -68.9432068, 46.8048210, -115.5079041, 115.5331573
17: -99.8758698, 53.5376968, -100.2169495, 53.8166809, -153.6925507, 153.7546387
18: -57.9355927, 49.1770592, -58.1565170, 49.2801094, -107.2156982, 107.3335724
19: -47.7652855, 34.9005051, -47.9074402, 34.9442215, -82.7095032, 82.8079453
20: -42.8812943, 36.5958939, -43.0384750, 36.6858253, -79.5671234, 79.6343689
21: -59.0250511, 40.1929512, -59.2466774, 40.3074875, -99.3325348, 99.4396286
22: -58.9576454, 40.2666626, -59.1544456, 40.3903580, -99.3480072, 99.4211121
23: -48.7801361, 40.0390244, -48.9006233, 40.1076317, -88.8877716, 88.9396362
24: -53.4054794, 39.4058075, -53.5958519, 39.5213127, -92.9267883, 93.0016556
25: -50.8674736, 45.3395767, -50.9866867, 45.4353333, -96.3028030, 96.3262482
26: -69.4961548, 61.0289116, -69.7340851, 61.2168732, -130.7130127, 130.7630005
27: -58.3303986, 40.5875931, -58.5609589, 40.7229309, -99.0533295, 99.1485519
28: -49.6492577, 44.8991508, -49.7632408, 44.9824791, -94.6317368, 94.6623840
29: -60.3825684, 35.0449524, -60.5234375, 35.1570625, -95.5396271, 95.5683899
30: -61.4099655, 44.6959343, -61.5401497, 44.8247604, -106.2347107, 106.2360840
31: -58.4612007, 39.5152512, -58.6570129, 39.5594177, -98.0206070, 98.1722641
32: -51.7687645, 39.3374176, -51.9424324, 39.4822884, -91.2510529, 91.2798462
33: -77.2641830, 49.9290657, -77.5125580, 50.1683426, -127.4325256, 127.4416199
34: -63.3065376, 42.1839752, -63.4645920, 42.3863144, -105.6928558, 105.6485672
35: -59.5916901, 42.0026512, -59.7834930, 42.2139435, -101.8056335, 101.7861404
36: -58.4514465, 42.8499374, -58.5925751, 42.9554062, -101.4068527, 101.4425125
37: -88.0665131, 50.1565704, -88.2998734, 50.2789230, -138.3454285, 138.4564514
38: -70.9414368, 52.5593109, -71.1222763, 52.7104340, -123.6518631, 123.6815872
39: -81.2605515, 44.1804886, -81.4559326, 44.2790451, -125.5395966, 125.6364212
40: -73.1261520, 39.5215302, -73.3369293, 39.6592979, -112.7854462, 112.8584595
41: -54.6799049, 43.9518509, -54.8372612, 44.0329895, -98.7128906, 98.7891083
42: -46.1449814, 39.5604744, -46.2992935, 39.7447128, -85.8896790, 85.8597717

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.5217369, upper bound: 78.2371467
time: 74.48 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.5278731, upper bound: 78.3829420
time: 73.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -74.5757980, 52.7078400, -74.3761292, 52.4770546, -127.0528564, 127.0839691
1: -44.0276184, 40.2635193, -43.8911934, 40.0977020, -84.1253204, 84.1547089
2: -40.9830551, 41.1415367, -40.8109283, 40.9598465, -81.9429016, 81.9524612
3: -49.6389465, 45.7026749, -49.4446793, 45.4511642, -95.0901031, 95.1473541
4: -51.4743156, 48.5084114, -51.2616539, 48.2496109, -99.7239075, 99.7700653
5: -50.7548676, 49.6516571, -50.5649643, 49.3937607, -100.1486282, 100.2166138
6: -57.3075066, 41.3689423, -57.1575584, 41.2588768, -98.5663834, 98.5265045
7: -56.1190834, 45.3534698, -55.9515572, 45.1959686, -101.3150482, 101.3050232
8: -56.9487076, 53.7119980, -56.7784424, 53.4970398, -110.4457474, 110.4904404
9: -51.1285706, 46.8353424, -50.9790726, 46.6987228, -97.8272934, 97.8144150
10: -67.1172638, 58.9838142, -66.8022156, 58.7511368, -125.8684006, 125.7860260
11: -65.1174774, 45.6111526, -64.7487183, 45.3718567, -110.4893341, 110.3598709
12: -60.3445625, 60.0511894, -59.9301949, 59.7777481, -120.1223068, 119.9813766
13: -63.8919258, 56.8364296, -63.8153572, 56.7126389, -120.6045532, 120.6517868
14: -98.5977325, 47.7745590, -98.2662506, 47.5449715, -146.1427002, 146.0408020
15: -59.8493042, 43.2038994, -59.6107826, 42.9895782, -102.8388672, 102.8146820
16: -68.6811981, 46.4174805, -68.4915619, 46.2386017, -114.9197998, 114.9090347
17: -99.8233795, 53.1538277, -99.4578705, 52.9271240, -152.7505035, 152.6116943
18: -57.8367386, 49.0735931, -57.6687202, 48.9706955, -106.8074341, 106.7423096
19: -47.7044029, 34.7396851, -47.5107307, 34.6017838, -82.3061829, 82.2504120
20: -42.8432922, 36.4064102, -42.6694336, 36.2754326, -79.1187210, 79.0758438
21: -59.0079002, 39.9400558, -58.7285156, 39.7445450, -98.7524414, 98.6685715
22: -58.9228859, 40.1297455, -58.8052940, 40.0005417, -98.9234314, 98.9350433
23: -48.7198410, 39.9172211, -48.5764236, 39.7926369, -88.5124817, 88.4936447
24: -53.2958336, 39.3903580, -53.2271309, 39.3596992, -92.6555176, 92.6174850
25: -50.7737732, 45.2316208, -50.6848984, 45.0975494, -95.8713226, 95.9165115
26: -69.4225845, 60.6711464, -69.1162262, 60.4480476, -129.8706055, 129.7873535
27: -58.2451744, 40.6040916, -58.1420021, 40.5455093, -98.7906799, 98.7460938
28: -49.6114426, 44.8349724, -49.4801292, 44.7150154, -94.3264618, 94.3151016
29: -60.3520050, 34.7889328, -60.1832962, 34.6407166, -94.9927139, 94.9722290
30: -61.3695068, 44.5102425, -61.2059669, 44.3379440, -105.7074432, 105.7162094
31: -58.3613434, 39.3945618, -58.2228851, 39.2675400, -97.6288757, 97.6174469
32: -51.7119102, 39.1084023, -51.5743752, 39.0072784, -90.7191696, 90.6827774
33: -77.0334930, 49.9434052, -76.8676147, 49.7525558, -126.7860184, 126.8110199
34: -63.1746445, 42.2032204, -63.0188942, 41.9980621, -105.1726990, 105.2221069
35: -59.4945335, 42.0641785, -59.3909149, 41.9035950, -101.3981323, 101.4550934
36: -58.4415512, 42.8227768, -58.3454056, 42.7344017, -101.1759491, 101.1681824
37: -88.0133514, 50.0947342, -87.8663483, 49.9585304, -137.9718628, 137.9610901
38: -70.8225021, 52.5743027, -70.6928711, 52.4561348, -123.2786407, 123.2671738
39: -81.1362228, 44.1489258, -80.9995728, 44.0165939, -125.1528168, 125.1484909
40: -73.0581665, 39.5330238, -72.9163742, 39.3937836, -112.4519501, 112.4494019
41: -54.6370888, 43.8380737, -54.5218582, 43.7453842, -98.3824768, 98.3599319
42: -46.1037331, 39.3364944, -45.9484024, 39.2082672, -85.3119965, 85.2848969

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2480418, upper bound: 78.2879683
time: 99.90 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2879684
time: 80.27 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -74.7156830, 52.7508392, -74.6914291, 52.6651115, -127.3807831, 127.4422684
1: -44.1134071, 40.2868767, -44.0920715, 40.2305908, -84.3439941, 84.3789520
2: -41.1085892, 41.1697159, -41.0803375, 41.1273499, -82.2359390, 82.2500534
3: -49.7892036, 45.7453003, -49.7592545, 45.6681900, -95.4573975, 95.5045471
4: -51.6590576, 48.5406990, -51.6429367, 48.4728394, -100.1318817, 100.1836395
5: -50.8922272, 49.7015991, -50.8524895, 49.6465149, -100.5387421, 100.5540924
6: -57.3597565, 41.4946938, -57.3600197, 41.5358696, -98.8956299, 98.8547134
7: -56.2097054, 45.3926773, -56.1781464, 45.3276978, -101.5373993, 101.5708160
8: -57.0677605, 53.7534370, -57.0413170, 53.7098389, -110.7775879, 110.7947540
9: -51.1941948, 46.9459152, -51.1455040, 46.9633217, -98.1575165, 98.0914154
10: -67.1837006, 59.1653786, -67.0502625, 59.1631966, -126.3468857, 126.2156372
11: -65.1802521, 45.8389587, -65.1139526, 45.8395309, -111.0197830, 110.9529114
12: -60.4035606, 60.3376465, -60.3238335, 60.3711853, -120.7747421, 120.6614838
13: -63.9424782, 56.9363403, -63.9938965, 56.9543648, -120.8968353, 120.9302368
14: -98.6813049, 48.0414352, -98.6266174, 48.0918655, -146.7731628, 146.6680603
15: -60.0712166, 43.2473488, -60.0754890, 43.2482262, -103.3194427, 103.3228378
16: -68.7467804, 46.5675850, -68.7254944, 46.5731850, -115.3199615, 115.2930756
17: -99.8962708, 53.4225769, -99.8955994, 53.4847641, -153.3810425, 153.3181763
18: -57.9408188, 49.1315308, -57.9243393, 49.1281815, -107.0689926, 107.0558701
19: -47.7556114, 34.8445816, -47.7462502, 34.8232574, -82.5788727, 82.5908356
20: -42.8891411, 36.5302505, -42.8604507, 36.5383911, -79.4275360, 79.3907013
21: -59.0623245, 40.0952911, -59.0134697, 40.0758438, -99.1381683, 99.1087646
22: -59.0069695, 40.1970291, -59.0344734, 40.1853561, -99.1923218, 99.2314987
23: -48.7663002, 40.0045891, -48.7610550, 39.9856949, -88.7519836, 88.7656403
24: -53.3909836, 39.4154701, -53.4510536, 39.4569473, -92.8479233, 92.8665237
25: -50.8392410, 45.3116379, -50.8718109, 45.2916679, -96.1309052, 96.1834488
26: -69.4900513, 60.8279572, -69.3880768, 60.7884941, -130.2785339, 130.2160339
27: -58.3477478, 40.6332474, -58.3826981, 40.6403885, -98.9881363, 99.0159454
28: -49.6534271, 44.9239273, -49.6600266, 44.9031982, -94.5566254, 94.5839539
29: -60.4032707, 34.9136086, -60.3969727, 34.9125137, -95.3157654, 95.3105774
30: -61.4176788, 44.6397438, -61.3783188, 44.6202240, -106.0378876, 106.0180588
31: -58.4281883, 39.4864311, -58.4364586, 39.4671135, -97.8953018, 97.9228897
32: -51.7598114, 39.2313423, -51.7826881, 39.2675514, -91.0273590, 91.0140305
33: -77.2000427, 49.9882317, -77.2628555, 49.9468575, -127.1468964, 127.2510834
34: -63.3034134, 42.2447357, -63.3073807, 42.2210236, -105.5244370, 105.5521164
35: -59.5891647, 42.0962219, -59.6193733, 42.0455971, -101.6347656, 101.7155914
36: -58.4892235, 42.8664932, -58.5058784, 42.8529663, -101.3421936, 101.3723755
37: -88.1143188, 50.1639252, -88.1524429, 50.1367416, -138.2510681, 138.3163757
38: -70.8961334, 52.6110802, -70.9199982, 52.5759506, -123.4720612, 123.5310822
39: -81.2344742, 44.1775856, -81.2834625, 44.1493378, -125.3838043, 125.4610443
40: -73.1545486, 39.5703964, -73.1681442, 39.5506554, -112.7052002, 112.7385406
41: -54.7083282, 43.9045334, -54.7182655, 43.9082718, -98.6166000, 98.6228027
42: -46.1552505, 39.4601402, -46.1568680, 39.4728165, -85.6280594, 85.6170044

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2900031
time: 64.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3812527
time: 74.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -74.8380127, 52.8631973, -74.4882736, 52.5133591, -127.3513489, 127.3514709
1: -44.1789322, 40.3543854, -43.9546471, 40.1193771, -84.2983093, 84.3090286
2: -41.2566071, 41.2856941, -40.9400253, 40.9820404, -82.2386475, 82.2257156
3: -49.9133720, 45.8715935, -49.5745621, 45.4895172, -95.4028778, 95.4461517
4: -51.8166695, 48.6509628, -51.4231453, 48.2812805, -100.0979462, 100.0740967
5: -51.0344315, 49.8718796, -50.6974030, 49.4349632, -100.4693909, 100.5692825
6: -57.4724388, 41.6038437, -57.2080727, 41.3636246, -98.8360596, 98.8119049
7: -56.3693123, 45.4700661, -56.0570717, 45.2271576, -101.5964661, 101.5271378
8: -57.1941719, 53.8734932, -56.8879929, 53.5317802, -110.7259521, 110.7614822
9: -51.2499123, 47.1417999, -51.0198708, 46.8341293, -98.0840454, 98.1616669
10: -67.3592072, 59.4999275, -66.8611755, 58.9872398, -126.3464508, 126.3610992
11: -65.3423615, 45.9660988, -64.7998810, 45.5397110, -110.8820724, 110.7659760
12: -60.6961670, 60.7088852, -59.9740410, 60.0952377, -120.7913971, 120.6829224
13: -64.0595245, 57.0823364, -63.8556061, 56.8119926, -120.8715210, 120.9379425
14: -98.9470749, 48.2694702, -98.3460770, 47.7840691, -146.7311401, 146.6155396
15: -60.1897888, 43.3842316, -59.7670174, 43.0304108, -103.2201996, 103.1512451
16: -68.8893280, 46.7131882, -68.5525818, 46.3685493, -115.2578659, 115.2657700
17: -100.1511536, 53.6373100, -99.5120773, 53.1566429, -153.3078003, 153.1493835
18: -58.0929527, 49.2133827, -57.7654686, 49.0237427, -107.1166916, 106.9788513
19: -47.8651924, 34.8209915, -47.5606842, 34.6356850, -82.5008698, 82.3816757
20: -43.0048943, 36.5733604, -42.7149811, 36.3506775, -79.3555756, 79.2883453
21: -59.2128410, 40.1439056, -58.7790680, 39.8360291, -99.0488586, 98.9229736
22: -59.1215019, 40.3128433, -58.8789406, 40.0698242, -99.1913300, 99.1917877
23: -48.8681145, 40.0095367, -48.6236916, 39.8280182, -88.6961212, 88.6332245
24: -53.5792274, 39.4993362, -53.3507500, 39.3809280, -92.9601440, 92.8500824
25: -50.9551239, 45.3462982, -50.7554016, 45.1363602, -96.0914841, 96.1016922
26: -69.6910706, 61.0267181, -69.1743469, 60.6140404, -130.3051147, 130.2010651
27: -58.5477028, 40.6962357, -58.2715721, 40.5663795, -99.1140747, 98.9678040
28: -49.7490425, 44.9010468, -49.5232544, 44.7413940, -94.4904327, 94.4242935
29: -60.5022011, 35.0162354, -60.2328262, 34.7442513, -95.2464523, 95.2490463
30: -61.5141983, 44.6919174, -61.2529297, 44.4148521, -105.9290466, 105.9448395
31: -58.6131516, 39.4603004, -58.3149643, 39.2915497, -97.9047012, 97.7752609
32: -51.9179459, 39.3830986, -51.6228256, 39.1365814, -91.0545197, 91.0059204
33: -77.4191208, 50.1264114, -77.0329208, 49.7937622, -127.2128677, 127.1593323
34: -63.3725090, 42.3486748, -63.0986328, 42.0311127, -105.4036255, 105.4473038
35: -59.7462959, 42.1874924, -59.4966164, 41.9330559, -101.6793518, 101.6841125
36: -58.5840187, 42.9106140, -58.3887253, 42.7668228, -101.3508301, 101.2993393
37: -88.2776947, 50.2097015, -87.9614716, 50.0005493, -138.2782288, 138.1711731
38: -71.0649185, 52.6801872, -70.7684555, 52.4906197, -123.5555420, 123.4486389
39: -81.3889236, 44.2498817, -81.0865707, 44.0444107, -125.4333267, 125.3364563
40: -73.3019867, 39.6278839, -73.0085907, 39.4157448, -112.7177277, 112.6364746
41: -54.8081017, 43.9815483, -54.5844536, 43.8041039, -98.6121979, 98.5660019
42: -46.2686157, 39.6355858, -45.9910202, 39.3455887, -85.6142044, 85.6266022

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3814459
time: 99.04 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3812527
time: 76.35 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -74.9812469, 52.9062996, -74.8061142, 52.7015190, -127.6827698, 127.7124100
1: -44.2686005, 40.3780136, -44.1575928, 40.2523460, -84.5209427, 84.5356064
2: -41.3833847, 41.3144341, -41.2105484, 41.1497879, -82.5331726, 82.5249786
3: -50.0647774, 45.9155312, -49.8899231, 45.7074165, -95.7721939, 95.8054428
4: -52.0045967, 48.6837082, -51.8069992, 48.5045319, -100.5091248, 100.4906998
5: -51.1731186, 49.9230156, -50.9860001, 49.6884537, -100.8615646, 100.9090118
6: -57.5268326, 41.7318611, -57.4116287, 41.6423683, -99.1692047, 99.1434784
7: -56.4631996, 45.5101471, -56.2860832, 45.3593254, -101.8225250, 101.7962341
8: -57.3165550, 53.9157257, -57.1535606, 53.7450256, -111.0615845, 111.0692825
9: -51.3167992, 47.2536926, -51.1868172, 47.0999069, -98.4167023, 98.4405060
10: -67.4270096, 59.6835327, -67.1094208, 59.4011688, -126.8281708, 126.7929535
11: -65.4068298, 46.1956635, -65.1661224, 46.0088348, -111.4156647, 111.3617859
12: -60.7560577, 60.9970436, -60.3680077, 60.6901398, -121.4461823, 121.3650436
13: -64.1115265, 57.1891937, -64.0352325, 57.0579643, -121.1694946, 121.2244186
14: -99.0324173, 48.5374374, -98.7074127, 48.3292389, -147.3616333, 147.2448425
15: -60.4147644, 43.4308815, -60.2340431, 43.2910233, -103.7057571, 103.6649246
16: -68.9594879, 46.8654518, -68.7896118, 46.7051620, -115.6646500, 115.6550598
17: -100.2252731, 53.9079018, -99.9505463, 53.7158432, -153.9411163, 153.8584442
18: -58.2050285, 49.2726288, -58.0281830, 49.1821213, -107.3871460, 107.3008118
19: -47.9182434, 34.9267426, -47.7968140, 34.8578796, -82.7761230, 82.7235565
20: -43.0517426, 36.6983109, -42.9066124, 36.6145134, -79.6662598, 79.6049194
21: -59.2685394, 40.3001480, -59.0649147, 40.1680908, -99.4366226, 99.3650589
22: -59.2073975, 40.3847618, -59.1095772, 40.2579079, -99.4653015, 99.4943314
23: -48.9159203, 40.0977859, -48.8084831, 40.0218315, -88.9377518, 88.9062653
24: -53.6776428, 39.5249863, -53.5772591, 39.4782944, -93.1559296, 93.1022491
25: -51.0224113, 45.4293747, -50.9438324, 45.3330879, -96.3554993, 96.3732071
26: -69.7599030, 61.1847382, -69.4472961, 60.9553871, -130.7152863, 130.6320190
27: -58.6559143, 40.7262192, -58.5164223, 40.6617203, -99.3176346, 99.2426453
28: -49.7921104, 44.9914169, -49.7035866, 44.9305801, -94.7226868, 94.6950073
29: -60.5549927, 35.1466789, -60.4471817, 35.0201378, -95.5751343, 95.5938568
30: -61.5642357, 44.8262444, -61.4264412, 44.7009354, -106.2651672, 106.2526855
31: -58.6851540, 39.5535011, -58.5312080, 39.4918594, -98.1770172, 98.0847092
32: -51.9667931, 39.5077095, -51.8317795, 39.3982620, -91.3650284, 91.3394852
33: -77.5874329, 50.1720238, -77.4296341, 49.9884109, -127.5758438, 127.6016541
34: -63.5027771, 42.3914108, -63.3881836, 42.2548409, -105.7576141, 105.7795944
35: -59.8428307, 42.2200546, -59.7265320, 42.0753860, -101.9182053, 101.9465790
36: -58.6326294, 42.9564095, -58.5497437, 42.8864746, -101.5191040, 101.5061493
37: -88.3847580, 50.2801514, -88.2516022, 50.1799622, -138.5647278, 138.5317535
38: -71.1396027, 52.7179031, -70.9965591, 52.6104240, -123.7500229, 123.7144623
39: -81.4902802, 44.2792091, -81.3729706, 44.1774521, -125.6677246, 125.6521759
40: -73.4054260, 39.6657715, -73.2659912, 39.5726128, -112.9780426, 112.9317627
41: -54.8826675, 44.0497284, -54.7828560, 43.9684830, -98.8511505, 98.8325806
42: -46.3199196, 39.7614479, -46.1990051, 39.6121140, -85.9320297, 85.9604492

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3816624
time: 71.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.5270371
time: 84.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -74.6950302, 52.7342644, -74.6628494, 52.7091789, -127.4042053, 127.3971024
1: -44.1060677, 40.2804451, -44.0776024, 40.2406464, -84.3467102, 84.3580475
2: -41.1249161, 41.1596031, -41.1221352, 41.1435394, -82.2684555, 82.2817383
3: -49.7871704, 45.7313995, -49.7662163, 45.6883354, -95.4755096, 95.4976120
4: -51.6327629, 48.5328255, -51.6150894, 48.4545364, -100.0872955, 100.1479111
5: -50.9068375, 49.6797943, -50.9025726, 49.6571350, -100.5639648, 100.5823593
6: -57.3546219, 41.4136658, -57.3197784, 41.3907967, -98.7454071, 98.7334442
7: -56.2329330, 45.3756561, -56.2383881, 45.3699417, -101.6028748, 101.6140442
8: -57.0724258, 53.7385674, -57.0620995, 53.6941986, -110.7666245, 110.8006668
9: -51.1660156, 46.9508095, -51.1436043, 46.9699364, -98.1359406, 98.0944061
10: -67.1747284, 59.2258072, -67.1769104, 59.2801208, -126.4548492, 126.4027176
11: -65.1563721, 45.7715759, -65.0273132, 45.7170906, -110.8734589, 110.7988892
12: -60.3763618, 60.3151360, -60.3511658, 60.3496933, -120.7260437, 120.6663055
13: -63.9291611, 56.9033241, -63.9289703, 56.9079704, -120.8371124, 120.8322830
14: -98.6671448, 47.9594116, -98.6588821, 47.9401321, -146.6072693, 146.6182861
15: -59.9834328, 43.2421494, -59.9219933, 43.1631775, -103.1466064, 103.1641388
16: -68.7348938, 46.5326805, -68.7079773, 46.5150185, -115.2499084, 115.2406540
17: -99.8660583, 53.2971115, -99.7758331, 53.2579193, -153.1239777, 153.0729370
18: -57.9032440, 49.1313019, -57.9117241, 49.1180649, -107.0213089, 107.0430298
19: -47.7424812, 34.7859383, -47.6776085, 34.7126122, -82.4550934, 82.4635468
20: -42.8824883, 36.4711380, -42.8532600, 36.4208298, -79.3033142, 79.3243942
21: -59.0468025, 40.0401764, -58.9702988, 39.9726257, -99.0194092, 99.0104752
22: -58.9680634, 40.1968994, -58.9628334, 40.1853333, -99.1533966, 99.1597290
23: -48.7530403, 39.9691849, -48.7194061, 39.9188843, -88.6719208, 88.6885910
24: -53.3548164, 39.4104080, -53.3866310, 39.4266281, -92.7814484, 92.7970352
25: -50.8069382, 45.2850647, -50.8001747, 45.2432785, -96.0502167, 96.0852356
26: -69.4672089, 60.8579636, -69.4745560, 60.8557663, -130.3229675, 130.3325195
27: -58.3279991, 40.6258850, -58.3584633, 40.6320114, -98.9600067, 98.9843445
28: -49.6451492, 44.8632965, -49.6033592, 44.7993050, -94.4444427, 94.4666595
29: -60.3871117, 34.8778725, -60.3279381, 34.8512497, -95.2383423, 95.2058105
30: -61.4014740, 44.6059380, -61.3759460, 44.5591316, -105.9606018, 105.9818802
31: -58.4187088, 39.4300842, -58.4392815, 39.3608780, -97.7795868, 97.8693695
32: -51.7533112, 39.1989441, -51.7501221, 39.2096062, -90.9629211, 90.9490662
33: -77.1545410, 49.9822159, -77.1483307, 49.9740524, -127.1285782, 127.1305466
34: -63.2526588, 42.2378120, -63.2118721, 42.1680679, -105.4207306, 105.4496765
35: -59.5900536, 42.0943260, -59.6046181, 42.0774269, -101.6674805, 101.6989441
36: -58.4890060, 42.8494186, -58.4788628, 42.8271637, -101.3161621, 101.3282776
37: -88.0751190, 50.1414337, -88.0631180, 50.1001282, -138.1752472, 138.2045593
38: -70.9073715, 52.6028061, -70.9234619, 52.5923843, -123.4997559, 123.5262680
39: -81.2016754, 44.1790352, -81.1919022, 44.1480370, -125.3497162, 125.3709412
40: -73.1236267, 39.5511971, -73.1153870, 39.5021973, -112.6258240, 112.6665649
41: -54.6907005, 43.8713074, -54.6755981, 43.8527603, -98.5434570, 98.5468903
42: -46.1409302, 39.4498558, -46.1079178, 39.4788475, -85.6197662, 85.5577698

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2948313
time: 70.30 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.4173939, upper bound: 78.3842472
time: 71.48 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -74.8348618, 52.7770844, -74.9806213, 52.8974762, -127.7323303, 127.7577057
1: -44.1916924, 40.3037224, -44.2805672, 40.3738327, -84.5655212, 84.5842896
2: -41.2504044, 41.1877365, -41.3923950, 41.3114243, -82.5618286, 82.5801315
3: -49.9373970, 45.7739296, -50.0823822, 45.9058533, -95.8432465, 95.8563080
4: -51.8173904, 48.5650253, -51.9976349, 48.6784172, -100.4958038, 100.5626602
5: -51.0441856, 49.7296638, -51.1913528, 49.9103546, -100.9545288, 100.9210129
6: -57.4067039, 41.5393562, -57.5236473, 41.6687813, -99.0754700, 99.0630035
7: -56.3237495, 45.4147720, -56.4685631, 45.5019417, -101.8256912, 101.8833313
8: -57.1913948, 53.7799454, -57.3263855, 53.9077568, -111.0991516, 111.1063309
9: -51.2315025, 47.0613403, -51.3105812, 47.2362823, -98.4677734, 98.3719177
10: -67.2410202, 59.4071045, -67.4252167, 59.6934624, -126.9344788, 126.8323212
11: -65.2190552, 45.9992905, -65.3933182, 46.1857986, -111.4048462, 111.3926086
12: -60.4352531, 60.6016541, -60.7455902, 60.9444580, -121.3796997, 121.3472366
13: -63.9796944, 57.0033035, -64.1086884, 57.1559258, -121.1356201, 121.1119919
14: -98.7505341, 48.2262955, -99.0204620, 48.4884949, -147.2390137, 147.2467651
15: -60.2052460, 43.2854156, -60.3885612, 43.4258041, -103.6310501, 103.6739731
16: -68.8001251, 46.6834526, -68.9482117, 46.8513222, -115.6514435, 115.6316681
17: -99.9388123, 53.5660934, -100.2143097, 53.8172951, -153.7561035, 153.7803955
18: -58.0085526, 49.1891060, -58.1729164, 49.2766495, -107.2852020, 107.3620148
19: -47.7937584, 34.8907318, -47.9132118, 34.9347992, -82.7285614, 82.8039398
20: -42.9282379, 36.5949440, -43.0443802, 36.6847305, -79.6129608, 79.6393280
21: -59.1010933, 40.1952705, -59.2561226, 40.3045387, -99.4056320, 99.4513931
22: -59.0521698, 40.2641830, -59.1930542, 40.3739319, -99.4261017, 99.4572296
23: -48.7993774, 40.0564346, -48.9046860, 40.1127434, -88.9121094, 88.9611206
24: -53.4504166, 39.4354820, -53.6142921, 39.5245438, -92.9749603, 93.0497665
25: -50.8724251, 45.3652344, -50.9876518, 45.4394951, -96.3119202, 96.3528900
26: -69.5345840, 61.0147743, -69.7469482, 61.1984291, -130.7330017, 130.7617188
27: -58.4307327, 40.6549492, -58.6036453, 40.7278748, -99.1586075, 99.2585907
28: -49.6870422, 44.9521790, -49.7839012, 44.9888458, -94.6758881, 94.7360764
29: -60.4383163, 35.0026169, -60.5424614, 35.1257248, -95.5640259, 95.5450745
30: -61.4495125, 44.7357292, -61.5490570, 44.8422089, -106.2917175, 106.2847824
31: -58.4855270, 39.5218582, -58.6578674, 39.5613594, -98.0468903, 98.1797256
32: -51.8010712, 39.3217850, -51.9591789, 39.4710274, -91.2720795, 91.2809601
33: -77.3211517, 50.0269394, -77.5452881, 50.1687164, -127.4898682, 127.5722198
34: -63.3815308, 42.2793350, -63.5020065, 42.3917198, -105.7732544, 105.7813416
35: -59.6846657, 42.1262283, -59.8345757, 42.2201233, -101.9047852, 101.9608002
36: -58.5366898, 42.8929977, -58.6402016, 42.9496307, -101.4863205, 101.5332031
37: -88.1761017, 50.2105484, -88.3518600, 50.2800331, -138.4561310, 138.5624084
38: -70.9811096, 52.6394348, -71.1518860, 52.7129097, -123.6940155, 123.7913208
39: -81.3000031, 44.2076530, -81.4775467, 44.2813568, -125.5813599, 125.6851959
40: -73.2202759, 39.5885506, -73.3707275, 39.6598434, -112.8801193, 112.9592743
41: -54.7617989, 43.9376068, -54.8737221, 44.0162277, -98.7780228, 98.8113174
42: -46.1922836, 39.5737991, -46.3167076, 39.7450371, -85.9373169, 85.8904953

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2956881
time: 69.43 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3842472
time: 75.12 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -74.9576874, 52.8893585, -74.7752762, 52.7451859, -127.7028656, 127.6646271
1: -44.2579994, 40.3710938, -44.1412849, 40.2620773, -84.5200806, 84.5123749
2: -41.3986130, 41.3036652, -41.2513466, 41.1655388, -82.5641479, 82.5550079
3: -50.0618362, 45.9002457, -49.8962708, 45.7263336, -95.7881699, 95.7965164
4: -51.9753151, 48.6753693, -51.7765770, 48.4860458, -100.4613495, 100.4519424
5: -51.1865692, 49.9000473, -51.0354080, 49.6980095, -100.8845749, 100.9354553
6: -57.5197372, 41.6490746, -57.3699112, 41.4954681, -99.0152054, 99.0189819
7: -56.4817581, 45.4921951, -56.3418694, 45.4007225, -101.8824768, 101.8340607
8: -57.3183212, 53.9000435, -57.1718102, 53.7287445, -111.0470581, 111.0718536
9: -51.2868423, 47.2574310, -51.1837997, 47.1054420, -98.3922653, 98.4412231
10: -67.4165192, 59.7418022, -67.2353973, 59.5162277, -126.9327469, 126.9771881
11: -65.3812256, 46.1263771, -65.0780716, 45.8848152, -111.2660370, 111.2044525
12: -60.7279510, 60.9731903, -60.3947563, 60.6672859, -121.3952255, 121.3679504
13: -64.0967484, 57.1504860, -63.9689255, 57.0075760, -121.1043243, 121.1194153
14: -99.0163727, 48.4547691, -98.7382507, 48.1792984, -147.1956635, 147.1930237
15: -60.3237228, 43.4227524, -60.0783501, 43.2037582, -103.5274734, 103.5010986
16: -68.9433899, 46.8286858, -68.7686539, 46.6455002, -115.5888901, 115.5973358
17: -100.1937943, 53.7811050, -99.8297424, 53.4877701, -153.6815643, 153.6108398
18: -58.1613121, 49.2710457, -58.0090752, 49.1710091, -107.3323212, 107.2801208
19: -47.9036217, 34.8672829, -47.7269554, 34.7466507, -82.6502686, 82.5942383
20: -43.0439606, 36.6381302, -42.8982620, 36.4960861, -79.5400391, 79.5363922
21: -59.2516785, 40.2440109, -59.0206146, 40.0641098, -99.3157654, 99.2646255
22: -59.1666794, 40.3808594, -59.0362930, 40.2548370, -99.4215164, 99.4171524
23: -48.9015579, 40.0614204, -48.7667122, 39.9544907, -88.8560486, 88.8281326
24: -53.6391068, 39.5192642, -53.5112991, 39.4476776, -93.0867767, 93.0305634
25: -50.9886131, 45.4000664, -50.8704948, 45.2821770, -96.2707901, 96.2705612
26: -69.7355804, 61.2137337, -69.5321198, 61.0222092, -130.7577820, 130.7458496
27: -58.6314201, 40.7180328, -58.4883652, 40.6527481, -99.2841644, 99.2063980
28: -49.7827339, 44.9294815, -49.6466217, 44.8256416, -94.6083755, 94.5761032
29: -60.5373306, 35.1043663, -60.3771133, 34.9527817, -95.4901123, 95.4814758
30: -61.5464745, 44.7877579, -61.4222183, 44.6359863, -106.1824417, 106.2099686
31: -58.6718826, 39.4958267, -58.5316048, 39.3847694, -98.0566406, 98.0274353
32: -51.9591637, 39.4737129, -51.7979927, 39.3389854, -91.2981491, 91.2717056
33: -77.5404129, 50.1649933, -77.3136444, 50.0147934, -127.5552063, 127.4786377
34: -63.4509697, 42.3833771, -63.2918015, 42.2008514, -105.6518097, 105.6751709
35: -59.8420982, 42.2174454, -59.7103462, 42.1066093, -101.9487076, 101.9277954
36: -58.6314850, 42.9376221, -58.5221176, 42.8595047, -101.4909744, 101.4597397
37: -88.3411179, 50.2564392, -88.1582336, 50.1420135, -138.4831238, 138.4146729
38: -71.1495819, 52.7086983, -70.9991608, 52.6265526, -123.7761383, 123.7078476
39: -81.4547119, 44.2799530, -81.2788544, 44.1757622, -125.6304626, 125.5588074
40: -73.3680191, 39.6460800, -73.2075958, 39.5241470, -112.8921661, 112.8536758
41: -54.8620567, 44.0147934, -54.7381363, 43.9114304, -98.7734833, 98.7529297
42: -46.3055687, 39.7496567, -46.1500969, 39.6167984, -85.9223633, 85.8997498

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2948313
time: 70.61 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.5278727
time: 73.57 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -75.1013718, 52.9325333, -75.0968094, 52.9338875, -128.0352631, 128.0293427
1: -44.3483963, 40.3948517, -44.3484573, 40.3957214, -84.7441101, 84.7433090
2: -41.5254669, 41.3325195, -41.5231171, 41.3340378, -82.8595047, 82.8556366
3: -50.2133865, 45.9443550, -50.2135506, 45.9454651, -96.1588364, 96.1579056
4: -52.1637192, 48.7081375, -52.1629639, 48.7102547, -100.8739624, 100.8710861
5: -51.3253441, 49.9513817, -51.3255463, 49.9525452, -101.2778778, 101.2769241
6: -57.5744400, 41.7771797, -57.5760460, 41.7756767, -99.3501129, 99.3532181
7: -56.5769653, 45.5324707, -56.5772171, 45.5336456, -102.1106110, 102.1096878
8: -57.4411392, 53.9424820, -57.4400711, 53.9434204, -111.3845520, 111.3825531
9: -51.3538284, 47.3694229, -51.3517265, 47.3733978, -98.7272034, 98.7211456
10: -67.4844894, 59.9253616, -67.4847412, 59.9320259, -127.4165039, 127.4100876
11: -65.4459534, 46.3560638, -65.4461060, 46.3554497, -111.8013916, 111.8021698
12: -60.7880821, 61.2615623, -60.7902260, 61.2639275, -122.0520096, 122.0517807
13: -64.1491699, 57.2589417, -64.1506424, 57.2629929, -121.4121628, 121.4095840
14: -99.1019974, 48.7233810, -99.1017227, 48.7267532, -147.8287506, 147.8251038
15: -60.5491524, 43.4699402, -60.5483551, 43.4700165, -104.0191498, 104.0182953
16: -69.0141296, 46.9819107, -69.0140915, 46.9845200, -115.9986420, 115.9960022
17: -100.2681122, 54.0521889, -100.2696991, 54.0494499, -154.3175659, 154.3218842
18: -58.2757568, 49.3304787, -58.2803955, 49.3310623, -107.6068192, 107.6108704
19: -47.9568329, 34.9730682, -47.9640541, 34.9698029, -82.9266357, 82.9371185
20: -43.0912704, 36.7632446, -43.0906906, 36.7612076, -79.8524780, 79.8539352
21: -59.3077087, 40.4003296, -59.3083344, 40.3971863, -99.7048645, 99.7086639
22: -59.2528992, 40.4536514, -59.2686157, 40.4486580, -99.7015381, 99.7222672
23: -48.9494133, 40.1497650, -48.9525719, 40.1495056, -89.0989227, 89.1023331
24: -53.7384186, 39.5450134, -53.7426071, 39.5459824, -93.2844009, 93.2876129
25: -51.0561256, 45.4838409, -51.0597763, 45.4822350, -96.5383606, 96.5436096
26: -69.8046722, 61.3720055, -69.8065186, 61.3663063, -131.1709747, 131.1785278
27: -58.7408600, 40.7481918, -58.7399445, 40.7495804, -99.4904404, 99.4881363
28: -49.8259010, 45.0200958, -49.8279762, 45.0168495, -94.8427429, 94.8480682
29: -60.5902939, 35.2360458, -60.5930824, 35.2340775, -95.8243561, 95.8291168
30: -61.5967369, 44.9225121, -61.5970802, 44.9233093, -106.5200348, 106.5195923
31: -58.7445145, 39.5892715, -58.7546463, 39.5866394, -98.3311462, 98.3439178
32: -52.0081062, 39.5983925, -52.0082817, 39.6021500, -91.6102448, 91.6066742
33: -77.7091217, 50.2106972, -77.7127380, 50.2102623, -127.9193878, 127.9234314
34: -63.5815620, 42.4263268, -63.5835457, 42.4258270, -106.0073853, 106.0098648
35: -59.9389191, 42.2500610, -59.9424591, 42.2500839, -102.1889954, 102.1925201
36: -58.6803589, 42.9837799, -58.6845436, 42.9840317, -101.6643753, 101.6683197
37: -88.4494324, 50.3268967, -88.4536209, 50.3233604, -138.7727814, 138.7805176
38: -71.2247162, 52.7464676, -71.2295151, 52.7474823, -123.9721680, 123.9759827
39: -81.5566254, 44.3093300, -81.5681076, 44.3094673, -125.8660889, 125.8774414
40: -73.4727936, 39.6841660, -73.4711761, 39.6821823, -113.1549759, 113.1553421
41: -54.9373169, 44.0830002, -54.9401627, 44.0767822, -99.0140991, 99.0231628
42: -46.3567810, 39.8760262, -46.3586655, 39.8853149, -86.2420959, 86.2346954

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1672

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3844853
time: 81.48 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.2465737, upper bound: 78.5278727
time: 83.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 167.48 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.1339610
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2311071
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.1361027
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2311685
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2320601
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3814815
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2325801
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3814815
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.1437883
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2365052
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.1453344
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.5222625, upper bound: 78.2365164
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2369164
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3829420
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.5217369, upper bound: 78.2371467
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.5278731, upper bound: 78.3829420
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2480418, upper bound: 78.2879683
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2879684
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2900031
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3812527
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3814459
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3812527
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3816624
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.5270371
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2948313
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.4173939, upper bound: 78.3842472
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2956881
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3842472
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2948313
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.5278727
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3844853
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 167.48
Output dim: 12, lower bound: -78.2465737, upper bound: 78.5278727

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -74.0588989, 52.3645782, -74.1548538, 52.4055862, -126.4644852, 126.5194321
1: -43.7172813, 40.0588837, -43.7614861, 40.0546913, -83.7719574, 83.8203659
2: -40.4348145, 40.8579063, -40.5580063, 40.9159393, -81.3507538, 81.4159088
3: -49.1020279, 45.3405609, -49.2010727, 45.3774109, -94.4794388, 94.5416336
4: -50.8028641, 48.1896095, -50.9434280, 48.1841812, -98.9870453, 99.1330414
5: -50.2087593, 49.2372665, -50.3216400, 49.3187714, -99.5275269, 99.5589066
6: -56.9692612, 41.0389824, -57.0468903, 41.1214371, -98.0906982, 98.0858765
7: -55.6300278, 45.0953522, -55.7572670, 45.1348801, -100.7649002, 100.8526154
8: -56.4429131, 53.3923264, -56.5538216, 53.4262085, -109.8691254, 109.9461517
9: -50.8239403, 46.2490463, -50.9003410, 46.4337463, -97.2576904, 97.1493835
10: -66.4721069, 57.8939438, -66.6777191, 58.2377014, -124.7097778, 124.5716553
11: -64.5704041, 44.8791237, -64.6508408, 45.0219727, -109.5923767, 109.5299683
12: -59.6284866, 58.8722687, -59.8472900, 59.2169762, -118.8454590, 118.7195511
13: -63.5915718, 56.3797989, -63.7344322, 56.5318222, -120.1233902, 120.1142273
14: -97.8707199, 46.8388290, -98.1187592, 47.0976028, -144.9683228, 144.9575806
15: -59.1933403, 42.8394890, -59.3153419, 42.9013329, -102.0946655, 102.1548233
16: -68.2213669, 45.8305206, -68.3728867, 45.9758530, -114.1972198, 114.2034073
17: -99.1832504, 52.3386803, -99.3625183, 52.5536575, -151.7368774, 151.7012024
18: -57.3366547, 48.7764969, -57.4972343, 48.8569260, -106.1935806, 106.2737274
19: -47.3771095, 34.5637817, -47.4176140, 34.5292664, -81.9063721, 81.9813919
20: -42.4921074, 36.1029930, -42.5810966, 36.1362076, -78.6283112, 78.6840897
21: -58.5460777, 39.5234604, -58.6321335, 39.5578384, -98.1039124, 98.1555939
22: -58.5111656, 39.7785263, -58.6564598, 39.8724480, -98.3836136, 98.4349823
23: -48.4377937, 39.7026215, -48.4940224, 39.7046700, -88.1424561, 88.1966400
24: -52.8422737, 39.1994553, -53.0313530, 39.3162537, -92.1585159, 92.2307968
25: -50.4966965, 44.9748459, -50.5816650, 45.0048943, -95.5015869, 95.5565109
26: -68.8480225, 59.9641228, -69.0044708, 60.1291161, -128.9771423, 128.9685974
27: -57.6856499, 40.3957520, -57.9055138, 40.4975510, -98.1831970, 98.3012695
28: -49.3196564, 44.6269722, -49.3857689, 44.6417389, -93.9613953, 94.0127335
29: -60.0250168, 34.3874588, -60.0833397, 34.4654922, -94.4904938, 94.4707947
30: -61.0659676, 44.0977859, -61.1204109, 44.1527328, -105.2187042, 105.2181931
31: -57.9227371, 39.2474899, -58.0760422, 39.2103729, -97.1331024, 97.3235321
32: -51.3267403, 38.6648636, -51.4729538, 38.7998962, -90.1266174, 90.1378098
33: -76.3482971, 49.5063133, -76.5594177, 49.6724892, -126.0207825, 126.0657349
34: -62.7619362, 41.8595963, -62.8425980, 41.9275284, -104.6894608, 104.7021866
35: -58.9891167, 41.7243805, -59.1606293, 41.8400040, -100.8291168, 100.8850098
36: -58.1227837, 42.5988541, -58.2257805, 42.6680679, -100.7908478, 100.8246307
37: -87.4778900, 49.7812576, -87.6648254, 49.8538284, -137.3317261, 137.4460754
38: -70.3898621, 52.3098488, -70.5334320, 52.3926811, -122.7825317, 122.8432770
39: -80.6957397, 43.9241486, -80.8338470, 43.9584694, -124.6542053, 124.7579956
40: -72.5814056, 39.3104515, -72.7377167, 39.3547783, -111.9361877, 112.0481644
41: -54.2632408, 43.5909882, -54.3761559, 43.6530838, -97.9163208, 97.9671478
42: -45.7579956, 38.8065071, -45.8565559, 38.9710960, -84.7290878, 84.6630630

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843869, upper bound: 78.1217600
time: 94.48 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.1280494
time: 68.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -74.2890320, 52.5228043, -74.2595596, 52.4256821, -126.7147064, 126.7823639
1: -43.8604813, 40.1469574, -43.8271484, 40.0672302, -83.9277039, 83.9740982
2: -40.6986732, 40.9904251, -40.6841202, 40.9291573, -81.6278305, 81.6745453
3: -49.3768082, 45.5093880, -49.3320885, 45.3998337, -94.7766418, 94.8414764
4: -51.0916176, 48.3258514, -51.0812759, 48.2028122, -99.2944336, 99.4071274
5: -50.4885635, 49.4283142, -50.4544144, 49.3408318, -99.8293915, 99.8827286
6: -57.0864296, 41.1467743, -57.0821838, 41.1654930, -98.2519150, 98.2289581
7: -55.8608398, 45.2055397, -55.8598022, 45.1517105, -101.0125504, 101.0653305
8: -56.6718292, 53.5194626, -56.6605339, 53.4453964, -110.1172256, 110.1799927
9: -50.9096603, 46.4633217, -50.9320564, 46.5255814, -97.4352341, 97.3953781
10: -66.6634064, 58.2822571, -66.7233734, 58.4143181, -125.0777283, 125.0056305
11: -64.7585297, 45.1649818, -64.6853790, 45.1594734, -109.9179840, 109.8503571
12: -59.9147682, 59.3619499, -59.8727150, 59.4523315, -119.3670959, 119.2346649
13: -63.6725197, 56.5174789, -63.7641640, 56.5815544, -120.2540665, 120.2816467
14: -98.1174927, 47.1502533, -98.1782303, 47.2451553, -145.3626404, 145.3284912
15: -59.4460106, 42.9577904, -59.4326019, 42.9313469, -102.3773575, 102.3903885
16: -68.3671112, 46.0042915, -68.4174881, 46.0485344, -114.4156494, 114.4217834
17: -99.4262314, 52.6273880, -99.4025040, 52.6877899, -152.1139984, 152.0298920
18: -57.4986649, 48.9119263, -57.5457535, 48.9154587, -106.4141235, 106.4576797
19: -47.5093117, 34.6608505, -47.4493828, 34.5731468, -82.0824509, 82.1102295
20: -42.6270103, 36.2320862, -42.6140137, 36.1963196, -78.8233261, 78.8460999
21: -58.7197151, 39.7204666, -58.6636772, 39.6490097, -98.3687286, 98.3841400
22: -58.6287193, 39.9300613, -58.6906166, 39.9342308, -98.5629425, 98.6206818
23: -48.5479050, 39.8055801, -48.5216751, 39.7503510, -88.2982559, 88.3272552
24: -52.9599800, 39.2488708, -53.0770149, 39.3338470, -92.2938156, 92.3258820
25: -50.5825424, 45.0805397, -50.6085434, 45.0478783, -95.6304169, 95.6890869
26: -69.1098633, 60.3130417, -69.0401764, 60.2956314, -129.4054871, 129.3532104
27: -57.8300362, 40.4406433, -57.9591217, 40.5157623, -98.3457947, 98.3997650
28: -49.4327126, 44.7097092, -49.4139252, 44.6783333, -94.1110458, 94.1236343
29: -60.1418037, 34.5844040, -60.1111946, 34.5594521, -94.7012558, 94.6955948
30: -61.1782722, 44.2765541, -61.1456108, 44.2354050, -105.4136810, 105.4221497
31: -58.0766830, 39.3154221, -58.1223488, 39.2380791, -97.3147583, 97.4377747
32: -51.4678955, 38.8411331, -51.5051537, 38.8828545, -90.3507538, 90.3462830
33: -76.5780029, 49.6557312, -76.6614838, 49.7074547, -126.2854538, 126.3172150
34: -62.8945656, 41.9560966, -62.8968277, 41.9557076, -104.8502655, 104.8529205
35: -59.1415749, 41.8127747, -59.2281189, 41.8645859, -101.0061646, 101.0408936
36: -58.2100067, 42.6825562, -58.2520981, 42.7019882, -100.9119873, 100.9346542
37: -87.6306839, 49.9163322, -87.7116928, 49.9096413, -137.5403290, 137.6280212
38: -70.5363007, 52.3825188, -70.5833740, 52.4161911, -122.9524689, 122.9658966
39: -80.8361969, 44.0162659, -80.8843536, 43.9836617, -124.8198547, 124.9006195
40: -72.7089386, 39.3680687, -72.7814789, 39.3690224, -112.0779572, 112.1495514
41: -54.3774147, 43.6913910, -54.4163284, 43.6934280, -98.0708389, 98.1077194
42: -45.8889771, 39.0299950, -45.8859100, 39.0722237, -84.9611969, 84.9158936

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2214432
time: 72.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.1280494
time: 75.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -74.1982193, 52.4080276, -74.4695435, 52.5941467, -126.7923584, 126.8775711
1: -43.8021011, 40.0825882, -43.9624634, 40.1880569, -83.9901581, 84.0450516
2: -40.5597992, 40.8864899, -40.8267899, 41.0837097, -81.6435013, 81.7132797
3: -49.2518578, 45.3836021, -49.5151024, 45.5951424, -94.8470001, 94.8986969
4: -50.9868851, 48.2221146, -51.3238258, 48.4078484, -99.3947296, 99.5459442
5: -50.3456841, 49.2880516, -50.6086540, 49.5721169, -99.9178009, 99.8967056
6: -57.0216751, 41.1638794, -57.2497368, 41.3975906, -98.4192429, 98.4136200
7: -55.7192001, 45.1347275, -55.9827194, 45.2668419, -100.9860382, 101.1174393
8: -56.5612259, 53.4342804, -56.8160172, 53.6395378, -110.2007599, 110.2502975
9: -50.8899498, 46.3588142, -51.0668488, 46.6976318, -97.5875854, 97.4256592
10: -66.5391235, 58.0744934, -66.9276657, 58.6488113, -125.1879272, 125.0021515
11: -64.6336670, 45.1063309, -65.0168762, 45.4891281, -110.1227951, 110.1232071
12: -59.6882820, 59.1580276, -60.2417603, 59.8093719, -119.4976425, 119.3997879
13: -63.6424332, 56.4791679, -63.9131241, 56.7730103, -120.4154434, 120.3922806
14: -97.9548874, 47.1087494, -98.4798660, 47.6453590, -145.6002502, 145.5886078
15: -59.4140892, 42.8836021, -59.7789764, 43.1608543, -102.5749207, 102.6625748
16: -68.2874451, 45.9792557, -68.6068497, 46.3085289, -114.5959778, 114.5861053
17: -99.2565765, 52.6066666, -99.8006744, 53.1101303, -152.3666840, 152.4073486
18: -57.4381752, 48.8344231, -57.7487297, 49.0144882, -106.4526672, 106.5831528
19: -47.4286728, 34.6686325, -47.6539116, 34.7507133, -82.1793823, 82.3225403
20: -42.5386238, 36.2266006, -42.7727509, 36.3989410, -78.9375610, 78.9993515
21: -58.6011200, 39.6784363, -58.9177361, 39.8888855, -98.4900055, 98.5961761
22: -58.5949326, 39.8450737, -58.8851585, 40.0565033, -98.6514359, 98.7302322
23: -48.4840851, 39.7896881, -48.6794472, 39.8975983, -88.3816833, 88.4691315
24: -52.9364777, 39.2247200, -53.2543945, 39.4137421, -92.3502197, 92.4791107
25: -50.5615692, 45.0542679, -50.7682343, 45.1979485, -95.7595215, 95.8224945
26: -68.9163284, 60.1204262, -69.2768936, 60.4689178, -129.3852386, 129.3973236
27: -57.7870636, 40.4252090, -58.1452866, 40.5927505, -98.3798141, 98.5704956
28: -49.3614159, 44.7160454, -49.5662918, 44.8298035, -94.1912231, 94.2823334
29: -60.0758934, 34.5116653, -60.2973671, 34.7366333, -94.8125305, 94.8090363
30: -61.1152534, 44.2259102, -61.2929459, 44.4335365, -105.5487823, 105.5188522
31: -57.9890442, 39.3391876, -58.2901573, 39.4098167, -97.3988571, 97.6293488
32: -51.3752251, 38.7873611, -51.6816978, 39.0598564, -90.4350815, 90.4690552
33: -76.5143127, 49.5516281, -76.9540405, 49.8673592, -126.3816681, 126.5056610
34: -62.8902893, 41.9015656, -63.1306839, 42.1508560, -105.0411377, 105.0322495
35: -59.0831947, 41.7569199, -59.3885384, 41.9823456, -101.0655365, 101.1454620
36: -58.1702576, 42.6430397, -58.3862801, 42.7868500, -100.9571075, 101.0293198
37: -87.5789261, 49.8498306, -87.9505997, 50.0312576, -137.6101685, 137.8004303
38: -70.4627075, 52.3468971, -70.7603149, 52.5128746, -122.9755859, 123.1072083
39: -80.7940598, 43.9526901, -81.1175079, 44.0913086, -124.8853607, 125.0701981
40: -72.6769333, 39.3481255, -72.9879456, 39.5118370, -112.1887665, 112.3360672
41: -54.3343430, 43.6573944, -54.5729027, 43.8156433, -98.1499863, 98.2303009
42: -45.8104973, 38.9290466, -46.0657692, 39.2345276, -85.0450287, 84.9948120

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843869, upper bound: 78.1241564
time: 104.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.1241564
time: 72.18 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -74.4284439, 52.5660057, -74.5743332, 52.6141167, -127.0425568, 127.1403351
1: -43.9451675, 40.1706238, -44.0268326, 40.2005157, -84.1456833, 84.1974564
2: -40.8237610, 41.0189133, -40.9529762, 41.0968628, -81.9206238, 81.9718933
3: -49.5266953, 45.5522842, -49.6461563, 45.6173744, -95.1440582, 95.1984406
4: -51.2757454, 48.3581390, -51.4618111, 48.4264450, -99.7021866, 99.8199463
5: -50.6255569, 49.4788589, -50.7414474, 49.5939674, -100.2195282, 100.2203064
6: -57.1387177, 41.2717972, -57.2848663, 41.4418411, -98.5805588, 98.5566635
7: -55.9503822, 45.2448196, -56.0854950, 45.2836533, -101.2340393, 101.3303146
8: -56.7903671, 53.5612335, -56.9228020, 53.6585579, -110.4489288, 110.4840393
9: -50.9756546, 46.5733757, -51.0985069, 46.7896538, -97.7653046, 97.6718826
10: -66.7302628, 58.4630737, -66.9725189, 58.8256035, -125.5558624, 125.4355698
11: -64.8215103, 45.3922501, -65.0512695, 45.6266823, -110.4481888, 110.4435196
12: -59.9743042, 59.6477623, -60.2669716, 60.0449524, -120.0192566, 119.9147339
13: -63.7233009, 56.6169434, -63.9427299, 56.8228073, -120.5461121, 120.5596619
14: -98.2013092, 47.4166374, -98.5390472, 47.7903824, -145.9916992, 145.9556885
15: -59.6672134, 43.0018616, -59.8965874, 43.1906090, -102.8578186, 102.8984528
16: -68.4330750, 46.1533279, -68.6513672, 46.3818932, -114.8149414, 114.8046875
17: -99.4992828, 52.8956299, -99.8405075, 53.2446785, -152.7439575, 152.7361298
18: -57.6012383, 48.9697800, -57.7989235, 49.0729065, -106.6741257, 106.7686920
19: -47.5607719, 34.7657089, -47.6854897, 34.7945328, -82.3553009, 82.4511948
20: -42.6730881, 36.3557358, -42.8054504, 36.4590912, -79.1321716, 79.1611862
21: -58.7746048, 39.8755035, -58.9491196, 39.9800568, -98.7546539, 98.8246231
22: -58.7123489, 39.9968872, -58.9193764, 40.1185455, -98.8308945, 98.9162598
23: -48.5941734, 39.8927650, -48.7068710, 39.9432907, -88.5374527, 88.5996399
24: -53.0546837, 39.2740631, -53.3004303, 39.4312744, -92.4859543, 92.5744934
25: -50.6475601, 45.1604271, -50.7951775, 45.2413483, -95.8889008, 95.9555969
26: -69.1778412, 60.4694176, -69.3124008, 60.6355591, -129.8134003, 129.7818146
27: -57.9318085, 40.4700012, -58.1991119, 40.6108627, -98.5426559, 98.6691132
28: -49.4740219, 44.7988205, -49.5941086, 44.8664017, -94.3404236, 94.3929291
29: -60.1925964, 34.7088165, -60.3250732, 34.8307877, -95.0233765, 95.0338898
30: -61.2273254, 44.4050903, -61.3180161, 44.5166473, -105.7439651, 105.7231064
31: -58.1432343, 39.4071465, -58.3362694, 39.4375763, -97.5808105, 97.7434158
32: -51.5160789, 38.9636955, -51.7137375, 39.1428452, -90.6589203, 90.6774292
33: -76.7441559, 49.7008553, -77.0563202, 49.9020805, -126.6462402, 126.7571716
34: -63.0230637, 41.9979172, -63.1850510, 42.1790009, -105.2020645, 105.1829681
35: -59.2357521, 41.8451385, -59.4562035, 42.0068207, -101.2425690, 101.3013382
36: -58.2573700, 42.7263222, -58.4125519, 42.8201370, -101.0775070, 101.1388702
37: -87.7313538, 49.9853287, -87.9974518, 50.0874252, -137.8187561, 137.9827881
38: -70.6090393, 52.4195251, -70.8101807, 52.5363274, -123.1453705, 123.2297058
39: -80.9342651, 44.0448151, -81.1680450, 44.1165390, -125.0507965, 125.2128601
40: -72.8045273, 39.4057159, -73.0322571, 39.5261078, -112.3306274, 112.4379730
41: -54.4483185, 43.7579269, -54.6129036, 43.8560486, -98.3043671, 98.3708344
42: -45.9407921, 39.1527443, -46.0949020, 39.3357925, -85.2765656, 85.2476349

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2215205
time: 76.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2228906
time: 67.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -74.3128128, 52.5206490, -74.2674561, 52.4421616, -126.7549744, 126.7881012
1: -43.8620682, 40.1479874, -43.8242493, 40.0764427, -83.9384995, 83.9722366
2: -40.7026825, 41.0012779, -40.6857491, 40.9379425, -81.6406174, 81.6870270
3: -49.3727722, 45.5071068, -49.3309555, 45.4158974, -94.7886581, 94.8380585
4: -51.1358490, 48.3298492, -51.1031685, 48.2158737, -99.3517227, 99.4330139
5: -50.4854927, 49.4545937, -50.4549904, 49.3600502, -99.8455429, 99.9095840
6: -57.1285172, 41.2493896, -57.0968628, 41.2176628, -98.3461761, 98.3462524
7: -55.8704910, 45.2092133, -55.8605156, 45.1659927, -101.0364838, 101.0697250
8: -56.6798782, 53.5519295, -56.6620483, 53.4610214, -110.1408997, 110.2139740
9: -50.9421082, 46.5470963, -50.9412041, 46.5679550, -97.5100632, 97.4882965
10: -66.7120972, 58.3984108, -66.7367401, 58.4719391, -125.1840363, 125.1351471
11: -64.7912598, 45.2257118, -64.7020645, 45.1886940, -109.9799500, 109.9277802
12: -59.9782944, 59.5226212, -59.8912239, 59.5332298, -119.5115204, 119.4138489
13: -63.7553482, 56.6165619, -63.7744408, 56.6305237, -120.3858643, 120.3909988
14: -98.2175369, 47.3305016, -98.1985474, 47.3359795, -145.5535126, 145.5290527
15: -59.5239372, 43.0117912, -59.4699936, 42.9423943, -102.4663315, 102.4817810
16: -68.4195251, 46.1136856, -68.4338226, 46.1041641, -114.5236893, 114.5475082
17: -99.5069656, 52.8167191, -99.4165802, 52.7823067, -152.2892761, 152.2332916
18: -57.5803452, 48.9125137, -57.5909386, 48.9096375, -106.4899826, 106.5034485
19: -47.5349655, 34.6422577, -47.4677734, 34.5627975, -82.0977554, 82.1100311
20: -42.6522942, 36.2659760, -42.6266937, 36.2110062, -78.8632965, 78.8926620
21: -58.7497864, 39.7247124, -58.6827583, 39.6488953, -98.3986816, 98.4074707
22: -58.7007141, 39.9541779, -58.7285004, 39.9414139, -98.6421280, 98.6826706
23: -48.5834084, 39.7901688, -48.5411911, 39.7397614, -88.3231659, 88.3313522
24: -53.1132355, 39.3059692, -53.1535492, 39.3372650, -92.4505005, 92.4595108
25: -50.6724510, 45.0850143, -50.6514320, 45.0433884, -95.7158356, 95.7364502
26: -69.1052856, 60.3145638, -69.0627136, 60.2954712, -129.4007568, 129.3772736
27: -57.9769669, 40.4822845, -58.0330086, 40.5181923, -98.4951477, 98.5152893
28: -49.4553299, 44.6896362, -49.4288025, 44.6679001, -94.1232300, 94.1184387
29: -60.1704445, 34.6056900, -60.1324844, 34.5662270, -94.7366714, 94.7381668
30: -61.2068863, 44.2714157, -61.1672630, 44.2288742, -105.4357605, 105.4386749
31: -58.1640282, 39.3106918, -58.1673889, 39.2341995, -97.3982239, 97.4780807
32: -51.5292320, 38.9314651, -51.5211487, 38.9285011, -90.4577332, 90.4526138
33: -76.7266541, 49.6884537, -76.7238312, 49.7136307, -126.4402847, 126.4122849
34: -62.9547729, 42.0032806, -62.9215317, 41.9604721, -104.9152374, 104.9248123
35: -59.2336769, 41.8442421, -59.2653389, 41.8696060, -101.1032715, 101.1095810
36: -58.2607880, 42.6814041, -58.2687263, 42.7003975, -100.9611816, 100.9501190
37: -87.7311020, 49.8915558, -87.7585220, 49.8954926, -137.6265869, 137.6500854
38: -70.6238098, 52.4135666, -70.6078339, 52.4270020, -123.0508041, 123.0214005
39: -80.9409485, 44.0238495, -80.9204559, 43.9862442, -124.9271927, 124.9442978
40: -72.8183899, 39.4033356, -72.8292313, 39.3766098, -112.1949997, 112.2325668
41: -54.4287758, 43.7294922, -54.4382744, 43.7110939, -98.1398697, 98.1677704
42: -45.9197884, 39.0872269, -45.8991470, 39.1024590, -85.0222397, 84.9863739

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843869, upper bound: 78.2220043
time: 68.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2241526
time: 81.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -74.5483170, 52.6801414, -74.3725052, 52.4622726, -127.0105896, 127.0526428
1: -44.0121002, 40.2390900, -43.8915520, 40.0891151, -84.1012115, 84.1306305
2: -40.9670639, 41.1354599, -40.8120308, 40.9513702, -81.9184265, 81.9474945
3: -49.6493683, 45.6792374, -49.4623413, 45.4386063, -95.0879745, 95.1415710
4: -51.4256554, 48.4674301, -51.2414627, 48.2346039, -99.6602478, 99.7088928
5: -50.7666321, 49.6481247, -50.5880547, 49.3823166, -100.1489334, 100.2361755
6: -57.2477837, 41.3701935, -57.1325607, 41.2667656, -98.5145416, 98.5027466
7: -56.1118584, 45.3230400, -55.9647026, 45.1831055, -101.2949677, 101.2877426
8: -56.9101105, 53.6811295, -56.7689514, 53.4803467, -110.3904572, 110.4500809
9: -51.0316925, 46.7639694, -50.9731255, 46.6601639, -97.6918411, 97.7370911
10: -66.9042053, 58.7889748, -66.7822876, 58.6489372, -125.5531464, 125.5712585
11: -64.9827728, 45.5135956, -64.7366714, 45.3265610, -110.3093338, 110.2502670
12: -60.2681503, 60.0131454, -59.9168015, 59.7688065, -120.0369568, 119.9299469
13: -63.8403549, 56.7578812, -63.8046265, 56.6803284, -120.5206833, 120.5625076
14: -98.4671555, 47.6434784, -98.2581177, 47.4839706, -145.9511261, 145.9015961
15: -59.7810287, 43.1324158, -59.5881958, 42.9724159, -102.7534409, 102.7206116
16: -68.5700150, 46.2931976, -68.4785767, 46.1773186, -114.7473221, 114.7717743
17: -99.7531433, 53.1083794, -99.4565964, 52.9168854, -152.6700287, 152.5649719
18: -57.7485352, 49.0499649, -57.6406174, 48.9684067, -106.7169418, 106.6905670
19: -47.6687012, 34.7400589, -47.4997368, 34.6068420, -82.2755280, 82.2397919
20: -42.7895050, 36.3968811, -42.6596985, 36.2712822, -79.0607910, 79.0565796
21: -58.9242783, 39.9226608, -58.7143898, 39.7401733, -98.6644516, 98.6370544
22: -58.8214493, 40.1084480, -58.7628746, 40.0034790, -98.8249283, 98.8713226
23: -48.6950150, 39.8952637, -48.5688744, 39.7855682, -88.4805756, 88.4641418
24: -53.2365532, 39.3572044, -53.2000618, 39.3550987, -92.5916519, 92.5572586
25: -50.7614403, 45.1930275, -50.6789284, 45.0868378, -95.8482666, 95.8719482
26: -69.3762741, 60.6700249, -69.0986786, 60.4624786, -129.8387451, 129.7687073
27: -58.1261177, 40.5286903, -58.0876808, 40.5365486, -98.6626663, 98.6163712
28: -49.5693855, 44.7736435, -49.4569435, 44.7046623, -94.2740479, 94.2305756
29: -60.2904396, 34.8151474, -60.1605835, 34.6639023, -94.9543457, 94.9757309
30: -61.3208351, 44.4536018, -61.1925392, 44.3119125, -105.6327438, 105.6461334
31: -58.3220024, 39.3803596, -58.2139473, 39.2621765, -97.5841827, 97.5943069
32: -51.6724358, 39.1102676, -51.5536423, 39.0118179, -90.6842499, 90.6639023
33: -76.9594803, 49.8389053, -76.8263321, 49.7486763, -126.7081604, 126.6652374
34: -63.0893936, 42.1006279, -62.9760551, 41.9887619, -105.0781555, 105.0766830
35: -59.3888855, 41.9350815, -59.3332748, 41.8940964, -101.2829819, 101.2683563
36: -58.3514824, 42.7673035, -58.2954407, 42.7343407, -101.0858231, 101.0627441
37: -87.8902130, 50.0288086, -87.8061600, 49.9516602, -137.8418732, 137.8349609
38: -70.7742996, 52.4875298, -70.6582413, 52.4506569, -123.2249603, 123.1457672
39: -81.0852356, 44.1162643, -80.9713821, 44.0114441, -125.0966797, 125.0876465
40: -72.9498672, 39.4619331, -72.8734283, 39.3909454, -112.3408127, 112.3353577
41: -54.5453720, 43.8309479, -54.4787979, 43.7516747, -98.2970276, 98.3097458
42: -46.0518074, 39.3136444, -45.9285202, 39.2042656, -85.2560730, 85.2421646

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2214432
time: 79.31 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.1280494
time: 85.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -74.4531250, 52.5635071, -74.5831299, 52.6303406, -127.0834656, 127.1466370
1: -43.9469719, 40.1713104, -44.0242844, 40.2094727, -84.1564484, 84.1955872
2: -40.8282852, 41.0295792, -40.9552956, 41.1055450, -81.9338226, 81.9848785
3: -49.5231934, 45.5497513, -49.6456070, 45.6332970, -95.1564865, 95.1953506
4: -51.3205757, 48.3620300, -51.4846878, 48.4391632, -99.7597275, 99.8467178
5: -50.6233139, 49.5048370, -50.7428436, 49.6131096, -100.2364197, 100.2476807
6: -57.1806793, 41.3757324, -57.2995224, 41.4953232, -98.6759872, 98.6752548
7: -55.9582214, 45.2485428, -56.0851440, 45.2978821, -101.2561035, 101.3336868
8: -56.7989769, 53.5933838, -56.9251862, 53.6739693, -110.4729462, 110.5185623
9: -51.0083160, 46.6575012, -51.1076469, 46.8325653, -97.8408813, 97.7651443
10: -66.7787781, 58.5800972, -66.9856033, 58.8842010, -125.6629791, 125.5656891
11: -64.8542938, 45.4536743, -65.0675812, 45.6566772, -110.5109711, 110.5212555
12: -60.0372925, 59.8093376, -60.2849464, 60.1267853, -120.1640778, 120.0942841
13: -63.8058357, 56.7161636, -63.9529686, 56.8715324, -120.6773682, 120.6691208
14: -98.3015289, 47.5965652, -98.5593414, 47.8812027, -146.1827393, 146.1558990
15: -59.7453651, 43.0558014, -59.9346161, 43.2014084, -102.9467773, 102.9904022
16: -68.4857941, 46.2634125, -68.6679230, 46.4380264, -114.9238205, 114.9313354
17: -99.5800323, 53.0850716, -99.8545914, 53.3394470, -152.9194641, 152.9396515
18: -57.6841125, 48.9705200, -57.8459015, 49.0673065, -106.7514038, 106.8164215
19: -47.5866585, 34.7473526, -47.7037544, 34.7845306, -82.3711853, 82.4511032
20: -42.6985016, 36.3898697, -42.8180161, 36.4740639, -79.1725616, 79.2078857
21: -58.8041344, 39.8798141, -58.9679413, 39.9801521, -98.7842865, 98.8477554
22: -58.7848282, 40.0215111, -58.9576225, 40.1260414, -98.9108734, 98.9791336
23: -48.6299324, 39.8774338, -48.7261848, 39.9329605, -88.5628967, 88.6036148
24: -53.2086525, 39.3311386, -53.3777161, 39.4344978, -92.6431427, 92.7088470
25: -50.7382355, 45.1650887, -50.8387222, 45.2373428, -95.9755707, 96.0038147
26: -69.1725311, 60.4711800, -69.3347168, 60.6358185, -129.8083496, 129.8058929
27: -58.0795631, 40.5114670, -58.2738113, 40.6131439, -98.6927032, 98.7852707
28: -49.4972267, 44.7787590, -49.6088181, 44.8561783, -94.3533936, 94.3875732
29: -60.2214699, 34.7303314, -60.3462715, 34.8379440, -95.0594101, 95.0765991
30: -61.2562752, 44.4039154, -61.3396759, 44.5129623, -105.7692413, 105.7435837
31: -58.2314529, 39.4024162, -58.3805771, 39.4335785, -97.6650314, 97.7829895
32: -51.5775108, 39.0548439, -51.7298584, 39.1892853, -90.7667847, 90.7846985
33: -76.8929977, 49.7334518, -77.1189651, 49.9081421, -126.8011398, 126.8524094
34: -63.0832443, 42.0450325, -63.2098312, 42.1835327, -105.2667770, 105.2548676
35: -59.3282471, 41.8764229, -59.4938011, 42.0117607, -101.3400116, 101.3702164
36: -58.3080444, 42.7255554, -58.4290314, 42.8189659, -101.1270142, 101.1545868
37: -87.8324051, 49.9608574, -88.0447083, 50.0738564, -137.9062653, 138.0055542
38: -70.6959000, 52.4507332, -70.8345184, 52.5467834, -123.2426758, 123.2852478
39: -81.0392914, 44.0526962, -81.2045822, 44.1190605, -125.1583557, 125.2572784
40: -72.9157181, 39.4406662, -73.0814819, 39.5333176, -112.4490128, 112.5221481
41: -54.4996071, 43.7970428, -54.6348343, 43.8747368, -98.3743439, 98.4318771
42: -45.9713440, 39.2108955, -46.1075706, 39.3668823, -85.3382263, 85.3184662

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843869, upper bound: 78.2225694
time: 80.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.3498617, upper bound: 78.2245120
time: 75.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -74.6901779, 52.7232895, -74.6893005, 52.6506424, -127.3408203, 127.4125900
1: -44.0982513, 40.2627678, -44.0923462, 40.2222900, -84.3205414, 84.3551102
2: -41.0931282, 41.1642075, -41.0819206, 41.1192474, -82.2123718, 82.2461243
3: -49.8001709, 45.7228432, -49.7771873, 45.6566429, -95.4568024, 95.5000229
4: -51.6120071, 48.4999237, -51.6242104, 48.4580879, -100.0700989, 100.1241302
5: -50.9048042, 49.6990814, -50.8760910, 49.6359024, -100.5407104, 100.5751648
6: -57.3013191, 41.4972153, -57.3361511, 41.5450134, -98.8463287, 98.8333664
7: -56.2033958, 45.3629379, -56.1920128, 45.3153763, -101.5187683, 101.5549393
8: -57.0309906, 53.7232018, -57.0333519, 53.6937904, -110.7247620, 110.7565384
9: -51.0985641, 46.8750954, -51.1399269, 46.9253273, -98.0238800, 98.0150223
10: -66.9717865, 58.9715042, -67.0311966, 59.0619965, -126.0337830, 126.0027008
11: -65.0469131, 45.7422676, -65.1031570, 45.7950478, -110.8419647, 110.8454208
12: -60.3278809, 60.3004570, -60.3110237, 60.3628769, -120.6907425, 120.6114655
13: -63.8919296, 56.8617058, -63.9839592, 56.9243774, -120.8163071, 120.8456650
14: -98.5520782, 47.9106979, -98.6195450, 48.0281410, -146.5802002, 146.5302429
15: -60.0044365, 43.1790581, -60.0540161, 43.2328682, -103.2373047, 103.2330780
16: -68.6391373, 46.4442253, -68.7147675, 46.5124969, -115.1516342, 115.1589966
17: -99.8269424, 53.3778000, -99.8951111, 53.4751091, -153.3020477, 153.2729187
18: -57.8567657, 49.1086426, -57.8999443, 49.1265144, -106.9832611, 107.0085754
19: -47.7212448, 34.8454704, -47.7362556, 34.8287468, -82.5499878, 82.5817261
20: -42.8361816, 36.5213470, -42.8513298, 36.5347977, -79.3709793, 79.3726807
21: -58.9796524, 40.0783920, -59.0003395, 40.0718613, -99.0515060, 99.0787277
22: -58.9062653, 40.1787262, -58.9929428, 40.1902313, -99.0964966, 99.1716614
23: -48.7421684, 39.9830093, -48.7541847, 39.9790802, -88.7212524, 88.7371979
24: -53.3339653, 39.3827019, -53.4257126, 39.4525299, -92.7864838, 92.8084106
25: -50.8279762, 45.2751236, -50.8668327, 45.2823563, -96.1103363, 96.1419525
26: -69.4446411, 60.8272552, -69.3714981, 60.8032722, -130.2479095, 130.1987610
27: -58.2317848, 40.5585136, -58.3309898, 40.6318855, -98.8636627, 98.8894958
28: -49.6112862, 44.8636169, -49.6370735, 44.8935509, -94.5048218, 94.5006866
29: -60.3421860, 34.9433670, -60.3748894, 34.9383011, -95.2804718, 95.3182526
30: -61.3712044, 44.5868950, -61.3658180, 44.5968666, -105.9680710, 105.9527130
31: -58.3921432, 39.4729347, -58.4291039, 39.4621429, -97.8542786, 97.9020386
32: -51.7211571, 39.2343597, -51.7626724, 39.2730827, -90.9942322, 90.9970245
33: -77.1268463, 49.8844223, -77.2223816, 49.9435081, -127.0703583, 127.1068039
34: -63.2187881, 42.1431389, -63.2650795, 42.2124367, -105.4312134, 105.4082184
35: -59.4844246, 41.9676476, -59.5625839, 42.0365372, -101.5209656, 101.5302277
36: -58.3993874, 42.8126755, -58.4562340, 42.8534164, -101.2528000, 101.2689056
37: -87.9942017, 50.0989838, -88.0947800, 50.1304131, -138.1246185, 138.1937561
38: -70.8474274, 52.5251350, -70.8856659, 52.5706902, -123.4181213, 123.4107971
39: -81.1851730, 44.1453247, -81.2569656, 44.1444969, -125.3296509, 125.4022827
40: -73.0505142, 39.4996948, -73.1286163, 39.5480843, -112.5986023, 112.6283112
41: -54.6182213, 43.8990173, -54.6767311, 43.9156952, -98.5339203, 98.5757370
42: -46.1032066, 39.4382935, -46.1369934, 39.4696388, -85.5728378, 85.5752792

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.3719776
time: 95.17 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.3723088
time: 82.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -74.1769104, 52.3910179, -74.4393082, 52.6379166, -126.8148270, 126.8303223
1: -43.7954140, 40.0756531, -43.9473457, 40.1976013, -83.9930115, 84.0229950
2: -40.5732574, 40.8758392, -40.8650246, 41.0992737, -81.6725311, 81.7408600
3: -49.2438622, 45.3687820, -49.5164185, 45.6143456, -94.8582077, 94.8852005
4: -50.9607391, 48.2131805, -51.2959671, 48.3887749, -99.3495178, 99.5091476
5: -50.3556480, 49.2654572, -50.6520233, 49.5819550, -99.9375992, 99.9174805
6: -57.0154457, 41.0815125, -57.2087059, 41.2509270, -98.2663727, 98.2902222
7: -55.7429352, 45.1168442, -56.0428276, 45.3086853, -101.0516205, 101.1596680
8: -56.5662651, 53.4188957, -56.8363914, 53.6232719, -110.1895294, 110.2552872
9: -50.8612556, 46.3635750, -51.0644760, 46.7037582, -97.5650101, 97.4280548
10: -66.5300446, 58.1356773, -67.0527649, 58.7653694, -125.2954102, 125.1884308
11: -64.6095428, 45.0385895, -64.9302902, 45.3655930, -109.9751358, 109.9688797
12: -59.6596184, 59.1359138, -60.2674217, 59.7877769, -119.4473877, 119.4033279
13: -63.6273117, 56.4452362, -63.8463402, 56.7260017, -120.3533020, 120.2915726
14: -97.9395752, 47.0193634, -98.5112152, 47.4900322, -145.4296112, 145.5305786
15: -59.3237038, 42.8781128, -59.6230011, 43.0751991, -102.3988953, 102.5011139
16: -68.2745972, 45.9409180, -68.5888367, 46.2461090, -114.5207062, 114.5297470
17: -99.2250900, 52.4766541, -99.6797562, 52.8818970, -152.1069794, 152.1564026
18: -57.3991852, 48.8331299, -57.7381363, 49.0028114, -106.4019928, 106.5712509
19: -47.4151726, 34.6103172, -47.5858307, 34.6391449, -82.0543213, 82.1961517
20: -42.5314560, 36.1671448, -42.7656479, 36.2808151, -78.8122711, 78.9327927
21: -58.5847664, 39.6235275, -58.8738251, 39.7851868, -98.3699493, 98.4973450
22: -58.5549355, 39.8446732, -58.8126945, 40.0562553, -98.6111832, 98.6573563
23: -48.4704399, 39.7498703, -48.6370010, 39.8273621, -88.2977982, 88.3868713
24: -52.8972626, 39.2185555, -53.1876602, 39.3822479, -92.2795105, 92.4062119
25: -50.5285110, 45.0287819, -50.6975060, 45.1494408, -95.6779480, 95.7262802
26: -68.8922958, 60.1378517, -69.3628998, 60.5254440, -129.4177246, 129.5007477
27: -57.7670135, 40.4172287, -58.1206818, 40.5836220, -98.3506165, 98.5379028
28: -49.3529282, 44.6548309, -49.5081940, 44.7251587, -94.0780869, 94.1630173
29: -60.0584183, 34.4756126, -60.2276115, 34.6750183, -94.7334366, 94.7032242
30: -61.0982552, 44.1919365, -61.2910690, 44.3723068, -105.4705658, 105.4830017
31: -57.9787407, 39.2822571, -58.2915306, 39.3024521, -97.2811890, 97.5737915
32: -51.3677788, 38.7535667, -51.6488190, 39.0004501, -90.3682098, 90.4023895
33: -76.4680328, 49.5453568, -76.8386230, 49.8943634, -126.3623962, 126.3839722
34: -62.8390617, 41.8946762, -63.0344925, 42.0977974, -104.9368591, 104.9291687
35: -59.0833626, 41.7552261, -59.3728523, 42.0147629, -101.0981293, 101.1280823
36: -58.1689224, 42.6258316, -58.3575706, 42.7607269, -100.9296265, 100.9833984
37: -87.5374908, 49.8278885, -87.8591766, 49.9953690, -137.5328522, 137.6870728
38: -70.4730606, 52.3381805, -70.7604980, 52.5291214, -123.0021820, 123.0986786
39: -80.7600708, 43.9546738, -81.0245972, 44.0903854, -124.8504486, 124.9792633
40: -72.6440048, 39.3283463, -72.9342117, 39.4631653, -112.1071701, 112.2625580
41: -54.3156815, 43.6251068, -54.5288048, 43.7604980, -98.0761795, 98.1539154
42: -45.7951012, 38.9076347, -46.0160065, 39.2320213, -85.0271225, 84.9236450

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843869, upper bound: 78.1297542
time: 68.53 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.4017538, upper bound: 78.1389647
time: 75.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -74.4076080, 52.5489388, -74.5443878, 52.6577225, -127.0653076, 127.0933228
1: -43.9391861, 40.1636505, -44.0132904, 40.2099533, -84.1491394, 84.1769409
2: -40.8376083, 41.0083389, -40.9915161, 41.1124878, -81.9500961, 81.9998474
3: -49.5194893, 45.5375786, -49.6481552, 45.6365509, -95.1560364, 95.1857300
4: -51.2498817, 48.3493195, -51.4339828, 48.4072533, -99.6571350, 99.7833023
5: -50.6362114, 49.4564857, -50.7854271, 49.6038933, -100.2401047, 100.2419128
6: -57.1326294, 41.1894531, -57.2440147, 41.2955704, -98.4281845, 98.4334641
7: -55.9732552, 45.2270203, -56.1437263, 45.3252182, -101.2984772, 101.3707428
8: -56.7957573, 53.5458908, -56.9435883, 53.6423035, -110.4380646, 110.4894714
9: -50.9467621, 46.5783997, -51.0958366, 46.7960014, -97.7427521, 97.6742325
10: -66.7212219, 58.5246429, -67.0980835, 58.9424057, -125.6636276, 125.6227264
11: -64.7973328, 45.3248367, -64.9640579, 45.5033112, -110.3006363, 110.2888947
12: -59.9457283, 59.6261559, -60.2928123, 60.0235634, -119.9692917, 119.9189682
13: -63.7084427, 56.5829048, -63.8760185, 56.7759743, -120.4844208, 120.4589157
14: -98.1858673, 47.3309708, -98.5701904, 47.6376495, -145.8235168, 145.9011383
15: -59.5787926, 42.9962883, -59.7409439, 43.1049385, -102.6837311, 102.7372284
16: -68.4204865, 46.1217117, -68.6331787, 46.3241196, -114.7446060, 114.7548904
17: -99.4677429, 52.7657700, -99.7193680, 53.0162659, -152.4840088, 152.4851379
18: -57.5618553, 48.9690552, -57.7873802, 49.0617828, -106.6236267, 106.7564316
19: -47.5472679, 34.7077866, -47.6172180, 34.6833420, -82.2306061, 82.3249969
20: -42.6659012, 36.2964401, -42.7979507, 36.3409996, -79.0068970, 79.0943909
21: -58.7582092, 39.8207626, -58.9050674, 39.8765259, -98.6347351, 98.7258301
22: -58.6722374, 39.9968834, -58.8467140, 40.1183968, -98.7906342, 98.8435974
23: -48.5805702, 39.8531494, -48.6644669, 39.8734436, -88.4540100, 88.5176163
24: -53.0173302, 39.2679596, -53.2351303, 39.3997765, -92.4171066, 92.5030823
25: -50.6146240, 45.1346016, -50.7242088, 45.1926422, -95.8072662, 95.8588104
26: -69.1538849, 60.4877014, -69.3980255, 60.6928749, -129.8467407, 129.8857269
27: -57.9117546, 40.4621964, -58.1746826, 40.6017990, -98.5135498, 98.6368790
28: -49.4655762, 44.7377472, -49.5364838, 44.7616768, -94.2272491, 94.2742233
29: -60.1751862, 34.6721115, -60.2550850, 34.7675781, -94.9427567, 94.9271927
30: -61.2103958, 44.3712158, -61.3157578, 44.4551430, -105.6655273, 105.6869736
31: -58.1330872, 39.3505325, -58.3379478, 39.3303604, -97.4634476, 97.6884766
32: -51.5087128, 38.9303856, -51.6806221, 39.0836334, -90.5923462, 90.6110001
33: -76.6979980, 49.6944008, -76.9409485, 49.9287109, -126.6267090, 126.6353302
34: -62.9719543, 41.9905319, -63.0889816, 42.1254997, -105.0974579, 105.0795135
35: -59.2361526, 41.8434334, -59.4407005, 42.0389633, -101.2751160, 101.2841263
36: -58.2560844, 42.7097054, -58.3839989, 42.7944603, -101.0505447, 101.0937042
37: -87.6902237, 49.9634972, -87.9062195, 50.0511246, -137.7413483, 137.8697205
38: -70.6196213, 52.4109154, -70.8113174, 52.5522957, -123.1719208, 123.2222290
39: -80.9005127, 44.0466690, -81.0751190, 44.1152306, -125.0157394, 125.1217880
40: -72.7715302, 39.3858604, -72.9782867, 39.4771347, -112.2486649, 112.3641434
41: -54.4296265, 43.7257347, -54.5690689, 43.8006592, -98.2302856, 98.2947998
42: -45.9256172, 39.1328125, -46.0450401, 39.3347397, -85.2603607, 85.1778488

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2260649
time: 69.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2260649
time: 72.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -74.3161850, 52.4343567, -74.7557907, 52.8265686, -127.1427536, 127.1901321
1: -43.8800659, 40.0993652, -44.1494255, 40.3310699, -84.2111206, 84.2487793
2: -40.6981735, 40.9043846, -41.1346474, 41.2674103, -81.9655762, 82.0390320
3: -49.3936691, 45.4117432, -49.8320084, 45.8321686, -95.2258377, 95.2437515
4: -51.1446342, 48.2455902, -51.6771202, 48.6129379, -99.7575684, 99.9227066
5: -50.4925461, 49.3161812, -50.9402351, 49.8354263, -100.3279724, 100.2564087
6: -57.0678978, 41.2062721, -57.4125443, 41.5279121, -98.5958099, 98.6188202
7: -55.8322716, 45.1561470, -56.2708702, 45.4408035, -101.2730713, 101.4270020
8: -56.6844254, 53.4607773, -57.0994377, 53.8370895, -110.5215149, 110.5602112
9: -50.9271812, 46.4732895, -51.2313118, 46.9692535, -97.8964310, 97.7046051
10: -66.5970764, 58.3159180, -67.3023529, 59.1775665, -125.7746429, 125.6182709
11: -64.6727448, 45.2656937, -65.2965698, 45.8335190, -110.5062637, 110.5622635
12: -59.7192688, 59.4216042, -60.6622925, 60.3812408, -120.1005096, 120.0838852
13: -63.6781845, 56.5446854, -64.0257721, 56.9715614, -120.6497345, 120.5704575
14: -98.0236816, 47.2894211, -98.8729858, 48.0388451, -146.0625305, 146.1624146
15: -59.5445175, 42.9220390, -60.0881310, 43.3376694, -102.8821869, 103.0101700
16: -68.3406525, 46.0899506, -68.8279190, 46.5802689, -114.9209213, 114.9178696
17: -99.2983093, 52.7447853, -100.1183167, 53.4398193, -152.7381287, 152.8630981
18: -57.5022278, 48.8909760, -57.9938774, 49.1611366, -106.6633606, 106.8848572
19: -47.4666786, 34.7150536, -47.8218842, 34.8611946, -82.3278732, 82.5369415
20: -42.5779037, 36.2907104, -42.9571114, 36.5443039, -79.1222076, 79.2478180
21: -58.6397018, 39.7783966, -59.1598244, 40.1166611, -98.7563629, 98.9382172
22: -58.6387024, 39.9112930, -59.0420609, 40.2429352, -98.8816376, 98.9533539
23: -48.5167122, 39.8369370, -48.8227463, 40.0209579, -88.5376663, 88.6596832
24: -52.9917564, 39.2437744, -53.4140358, 39.4802284, -92.4719772, 92.6578064
25: -50.5934410, 45.1083946, -50.8844032, 45.3438950, -95.9373322, 95.9927979
26: -68.9604340, 60.2941513, -69.6352539, 60.8675041, -129.8279419, 129.9294128
27: -57.8687477, 40.4466095, -58.3640137, 40.6795502, -98.5482941, 98.8106155
28: -49.3945923, 44.7438126, -49.6891251, 44.9141998, -94.3087921, 94.4329376
29: -60.1092300, 34.5999146, -60.4421539, 34.9474030, -95.0566330, 95.0420609
30: -61.1473961, 44.3204041, -61.4641914, 44.6537132, -105.8011093, 105.7845840
31: -58.0449600, 39.3738861, -58.5098343, 39.5024567, -97.5474167, 97.8837204
32: -51.4161911, 38.8759193, -51.8580666, 39.2613564, -90.6775513, 90.7339859
33: -76.6341171, 49.5905952, -77.2345734, 50.0892563, -126.7233582, 126.8251648
34: -62.9675064, 41.9366226, -63.3239441, 42.3213615, -105.2888641, 105.2605591
35: -59.1774559, 41.7876167, -59.6020241, 42.1575890, -101.3350449, 101.3896408
36: -58.2163773, 42.6699448, -58.5186882, 42.8828239, -101.0991974, 101.1886292
37: -87.6385956, 49.8963699, -88.1463165, 50.1744156, -137.8130188, 138.0426941
38: -70.5460281, 52.3751297, -70.9884949, 52.6498909, -123.1958923, 123.3636246
39: -80.8583221, 43.9832535, -81.3092194, 44.2236176, -125.0819244, 125.2924728
40: -72.7397079, 39.3660736, -73.1866760, 39.6208038, -112.3605118, 112.5527496
41: -54.3868217, 43.6913681, -54.7263184, 43.9235382, -98.3103485, 98.4176788
42: -45.8474884, 39.0306206, -46.2255745, 39.4973412, -85.3448257, 85.2561874

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.4209129, upper bound: 78.1313924
time: 171.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.5092791, upper bound: 78.1399996
time: 68.17 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -74.5469437, 52.5920944, -74.8610535, 52.8462791, -127.3932190, 127.4531479
1: -44.0236893, 40.1872482, -44.2143402, 40.3433304, -84.3670197, 84.4015884
2: -40.9625931, 41.0367203, -41.2611771, 41.2805290, -82.2431183, 82.2978973
3: -49.6692734, 45.5803528, -49.9637375, 45.8542480, -95.5235214, 95.5440903
4: -51.4338684, 48.3815994, -51.8153763, 48.6313438, -100.0652161, 100.1969681
5: -50.7731590, 49.5069733, -51.0737076, 49.8572159, -100.6303711, 100.5806808
6: -57.1849594, 41.3143234, -57.4477730, 41.5730972, -98.7580490, 98.7621002
7: -56.0631332, 45.2661972, -56.3723335, 45.4573097, -101.5204468, 101.6385269
8: -56.9141197, 53.5875282, -57.2067680, 53.8560333, -110.7701569, 110.7942963
9: -51.0126495, 46.6884117, -51.2626953, 47.0617218, -98.0743561, 97.9511032
10: -66.7879562, 58.7051888, -67.3470459, 59.3548012, -126.1427536, 126.0522308
11: -64.8603058, 45.5519943, -65.3303146, 45.9713287, -110.8316345, 110.8823090
12: -60.0051727, 59.9119987, -60.6875038, 60.6172562, -120.6224136, 120.5995026
13: -63.7591171, 56.6823883, -64.0554581, 57.0222282, -120.7813416, 120.7378464
14: -98.2695923, 47.5975494, -98.9317856, 48.1840096, -146.4535980, 146.5293274
15: -59.7999687, 43.0402031, -60.2062798, 43.3673706, -103.1673431, 103.2464828
16: -68.4863052, 46.2714348, -68.8724060, 46.6590462, -115.1453476, 115.1438446
17: -99.5406952, 53.0341759, -100.1578140, 53.5746689, -153.1153564, 153.1919861
18: -57.6657219, 49.0268021, -58.0449524, 49.2200623, -106.8857880, 107.0717545
19: -47.5986061, 34.8124847, -47.8531456, 34.9053574, -82.5039597, 82.6656342
20: -42.7118797, 36.4200516, -42.9892616, 36.6045799, -79.3164597, 79.4093094
21: -58.8129730, 39.9756737, -59.1910400, 40.2080383, -99.0209961, 99.1667099
22: -58.7558899, 40.0638885, -59.0762253, 40.3056412, -99.0615311, 99.1401062
23: -48.6267014, 39.9402695, -48.8500519, 40.0670547, -88.6937561, 88.7903214
24: -53.1123810, 39.2931442, -53.4619141, 39.4977226, -92.6101074, 92.7550583
25: -50.6795959, 45.2145996, -50.9112015, 45.3876572, -96.0672531, 96.1258011
26: -69.2217407, 60.6440811, -69.6703796, 61.0350609, -130.2568054, 130.3144531
27: -58.0137177, 40.4915123, -58.4184380, 40.6976700, -98.7113876, 98.9099350
28: -49.5067863, 44.8267365, -49.7171097, 44.9508057, -94.4575958, 94.5438461
29: -60.2258797, 34.7966614, -60.4695358, 35.0404892, -95.2663651, 95.2661972
30: -61.2593079, 44.5000763, -61.4888344, 44.7370262, -105.9963226, 105.9889069
31: -58.1996155, 39.4421844, -58.5561867, 39.5304947, -97.7301025, 97.9983673
32: -51.5568161, 39.0527725, -51.8897247, 39.3445778, -90.9013901, 90.9424973
33: -76.8642120, 49.7393990, -77.3371964, 50.1234856, -126.9876938, 127.0765991
34: -63.1004868, 42.0323334, -63.3786125, 42.3490639, -105.4495544, 105.4109497
35: -59.3303566, 41.8756561, -59.6700668, 42.1816864, -101.5120316, 101.5457153
36: -58.3034554, 42.7533798, -58.5451431, 42.9160538, -101.2195129, 101.2985229
37: -87.7909546, 50.0323830, -88.1936646, 50.2304840, -138.0214386, 138.2260437
38: -70.6924591, 52.4478302, -71.0393524, 52.6730499, -123.3655090, 123.4871826
39: -80.9984665, 44.0751648, -81.3599548, 44.2484741, -125.2469406, 125.4351196
40: -72.8673706, 39.4234619, -73.2315063, 39.6348305, -112.5021973, 112.6549683
41: -54.5004730, 43.7921600, -54.7666283, 43.9638824, -98.4643555, 98.5587769
42: -45.9773560, 39.2560043, -46.2544518, 39.6002426, -85.5775909, 85.5104523

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1368

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.1313924
time: 247.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2299454
time: 80.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 330.89 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843869, upper bound: 78.1217600
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.1280494
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2214432
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.1280494
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843869, upper bound: 78.1241564
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.1241564
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2215205
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2228906
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843869, upper bound: 78.2220043
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2241526
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2214432
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.1280494
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843869, upper bound: 78.2225694
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.3498617, upper bound: 78.2245120
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.3719776
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.3723088
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843869, upper bound: 78.1297542
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.4017538, upper bound: 78.1389647
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2260649
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2260649
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.4209129, upper bound: 78.1313924
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.5092791, upper bound: 78.1399996
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.1313924
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 330.89
Output dim: 12, lower bound: -78.1843870, upper bound: 78.2299454
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2369164
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3829420
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.5217369, upper bound: 78.2371467
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.5278731, upper bound: 78.3829420
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2480418, upper bound: 78.2879683
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2879684
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2900031
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3812527
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3814459
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3812527
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3816624
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.5270371
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2948313
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.4173939, upper bound: 78.3842472
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2956881
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3842472
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.2948313
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.5278727
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.3844853
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 330.89
Output dim: 12, lower bound: -78.2465737, upper bound: 78.5278727

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 87.05 + 7120.94 = 7207.99 seconds
