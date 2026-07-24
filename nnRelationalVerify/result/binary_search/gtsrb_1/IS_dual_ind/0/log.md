## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 111.228024969
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505)
1: (-62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425)
2: (-58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823)
3: (-68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288)
4: (-73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358)
5: (-70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624)
6: (-81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562)
7: (-78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395)
8: (-81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184)
9: (-71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248)
10: (-98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989)
11: (-93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175)
12: (-87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857)
13: (-90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535)
14: (-133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552)
15: (-85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512)
16: (-101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865)
17: (-136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178)
18: (-85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041)
19: (-68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367)
20: (-61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456)
21: (-84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706)
22: (-85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717)
23: (-70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595)
24: (-79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519)
25: (-72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140)
26: (-98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664)
27: (-85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523)
28: (-69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379)
29: (-88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917)
30: (-87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264)
31: (-85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191)
32: (-75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396)
33: (-108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599)
34: (-87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909)
35: (-84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664)
36: (-82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775)
37: (-123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730)
38: (-101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500)
39: (-115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056)
40: (-98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000)
41: (-78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470)
42: (-63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209)

## BASE Result
execution time: IAR + LP analysis = 2.79 + 112.42 = 115.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -120.7499906, upper bound: 120.7499906


# Binary Search by BASE starts (time budget: 17884.80 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=177.0996856689453
rel_dist={12: [-115.71834210356894, 115.71834210370692]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=177.0996856689453
rel_dist={12: [-111.25869176025041, 111.25869175221396]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=177.0996856689453
rel_dist={12: [-107.1449640304723, 107.14496402430538]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=177.0996856689453
rel_dist={12: [-109.3687141926502, 109.36871419485921]}

## Binary Search Result
Binary search time: 498.35 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 17386.45 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

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
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8804127, upper bound: 116.6609588
time: 83.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8804127, upper bound: 116.8804124
time: 92.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 176.26 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 176.26
Output dim: 12, lower bound: -116.8804127, upper bound: 116.6609588
IS_A2, status: Status.UNKNOWN, split count: 1, time: 176.26
Output dim: 12, lower bound: -116.8804127, upper bound: 116.8804124

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -105.7478790, 76.8107834, -106.0832443, 77.0181427, -182.7660217, 182.8940277
1: -62.6895599, 57.5903893, -62.8877182, 57.7230301, -120.4125900, 120.4781036
2: -58.2942429, 58.0871048, -58.6151085, 58.2519226, -116.5461655, 116.7022018
3: -67.6614227, 66.6938324, -67.9680023, 66.9093628, -134.5707855, 134.6618347
4: -73.0185394, 69.1616516, -73.4222336, 69.3708496, -142.3893890, 142.5838928
5: -69.9693909, 71.5915070, -70.2843475, 71.8342285, -141.8036194, 141.8758545
6: -81.2805939, 61.1545334, -81.5494385, 61.4160690, -142.6966400, 142.7039642
7: -77.6139984, 64.8577576, -77.9275436, 65.0259247, -142.6399231, 142.7853088
8: -81.2684784, 78.5836639, -81.6048508, 78.7881393, -160.0566101, 160.1885071
9: -71.6087952, 68.1343994, -71.8496552, 68.5363617, -140.1451569, 139.9840393
10: -97.5931625, 85.9088516, -98.1051712, 86.6370010, -184.2301636, 184.0140228
11: -92.5323410, 65.1348572, -92.9693146, 65.6177521, -158.1500854, 158.1041718
12: -86.4826508, 89.0872192, -87.0006561, 89.8041153, -176.2867737, 176.0878601
13: -90.2257690, 97.9576340, -90.4919205, 98.4202194, -188.6459656, 188.4495544
14: -133.1739807, 75.6484756, -133.7793427, 76.3016968, -209.4756775, 209.4278259
15: -84.7683563, 63.9488258, -85.2105179, 64.2373352, -149.0056915, 149.1593323
16: -101.3795853, 68.7092209, -101.7267227, 69.1567688, -170.5363464, 170.4359131
17: -135.5813141, 81.3465424, -136.0882568, 81.8995743, -217.4808807, 217.4347839
18: -85.3504333, 68.8810425, -85.7657928, 69.0820923, -154.4325256, 154.6468353
19: -68.3848419, 48.4514236, -68.6249008, 48.5536842, -116.9385223, 117.0763245
20: -60.8479576, 52.7312737, -61.0831146, 52.9198265, -113.7677841, 113.8143768
21: -84.5057297, 57.6774330, -84.8353195, 57.9100609, -142.4157867, 142.5127563
22: -85.2066956, 58.0677567, -85.5708923, 58.3169594, -143.5236511, 143.6386414
23: -69.9317780, 57.3499985, -70.1545258, 57.4897423, -127.4214935, 127.5045242
24: -78.8860855, 52.8611717, -79.2627258, 53.0123940, -131.8984833, 132.1239014
25: -72.6233521, 62.0051537, -72.8486938, 62.1972771, -134.8206177, 134.8538513
26: -98.2069397, 87.1182098, -98.5726395, 87.5061646, -185.7130737, 185.6908264
27: -85.2542801, 58.8834457, -85.7265015, 59.0911140, -144.3453979, 144.6099548
28: -69.6882782, 63.5879440, -69.9148407, 63.7420158, -133.4302979, 133.5027771
29: -88.2226257, 50.9105492, -88.4925156, 51.1498451, -139.3724670, 139.4030457
30: -86.8901520, 65.1539383, -87.1247330, 65.4327469, -152.3229065, 152.2786713
31: -84.8343201, 55.4123650, -85.1798782, 55.5154533, -140.3497772, 140.5922241
32: -75.5650330, 61.7727699, -75.8054504, 62.0880775, -137.6530762, 137.5782166
33: -107.5151291, 82.1626663, -108.0078430, 82.4851227, -190.0002441, 190.1705017
34: -87.4266281, 65.5697861, -87.7541580, 65.8691025, -153.2957306, 153.3239441
35: -83.5157547, 68.3576660, -83.9027939, 68.6390305, -152.1547852, 152.2604675
36: -82.6186447, 73.5060120, -82.8912964, 73.6660614, -156.2846985, 156.3973083
37: -123.0927048, 71.0162354, -123.5434113, 71.2149048, -194.3075867, 194.5596466
38: -100.9975967, 92.9542847, -101.3370667, 93.1702881, -194.1678467, 194.2913513
39: -115.1795654, 83.3827515, -115.5448456, 83.5421066, -198.7216797, 198.9275970
40: -98.3746643, 60.0926590, -98.7978592, 60.3096695, -158.6843262, 158.8905182
41: -78.5238266, 62.8973999, -78.8389130, 63.0652847, -141.5891113, 141.7363129
42: -63.3596535, 58.6848717, -63.6166687, 59.0460968, -122.4057465, 122.3015442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=448, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=635, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6262029, upper bound: 116.6363803
time: 78.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6262029, upper bound: 116.6363803
time: 90.92 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -106.1341934, 77.0433197, -106.1664734, 77.0547638, -183.1889648, 183.2097778
1: -62.9125290, 57.7340736, -62.9334106, 57.7456436, -120.6581726, 120.6674805
2: -58.6830139, 58.2630463, -58.7094879, 58.2740669, -116.9570618, 116.9725342
3: -68.0199890, 66.9307556, -68.0529251, 66.9455261, -134.9655151, 134.9836731
4: -73.5132523, 69.3889694, -73.5506287, 69.4037476, -142.9169922, 142.9395752
5: -70.3380203, 71.8568115, -70.3715363, 71.8713684, -142.2093811, 142.2283478
6: -81.5817413, 61.4390564, -81.6051941, 61.4806366, -143.0623779, 143.0442505
7: -77.9623184, 65.0399628, -77.9943542, 65.0572205, -143.0195312, 143.0343170
8: -81.6675262, 78.8113861, -81.6988602, 78.8245087, -160.4920197, 160.5102539
9: -71.8656158, 68.6264420, -71.8823013, 68.6605225, -140.5261230, 140.5087433
10: -98.1372833, 86.8112640, -98.1612549, 86.8737335, -185.0109863, 184.9725189
11: -92.9973526, 65.7273712, -93.0149841, 65.7719879, -158.7693481, 158.7423553
12: -87.0223999, 89.9753723, -87.0406876, 90.0328064, -177.0552063, 177.0160522
13: -90.5028687, 98.5130844, -90.5254059, 98.5483780, -189.0512390, 189.0384827
14: -133.8108521, 76.4723053, -133.8418579, 76.5200195, -210.3308716, 210.3141632
15: -85.2975311, 64.2654114, -85.3384247, 64.2796783, -149.5771942, 149.6038361
16: -101.7602463, 69.2506409, -101.7859802, 69.2923431, -171.0525818, 171.0366211
17: -136.1045074, 82.0433655, -136.1282654, 82.0756454, -218.1801453, 218.1716309
18: -85.8267746, 69.0953674, -85.8637543, 69.1220322, -154.9488068, 154.9591064
19: -68.6331329, 48.5608444, -68.6683044, 48.5743828, -117.2075195, 117.2291489
20: -61.1089859, 52.9547348, -61.1209641, 52.9748688, -114.0838547, 114.0756989
21: -84.8646088, 57.9534798, -84.8830338, 57.9754715, -142.8400726, 142.8365173
22: -85.6066132, 58.3436966, -85.6569672, 58.3651619, -143.9717712, 144.0006714
23: -70.1796722, 57.4995766, -70.1979675, 57.5214272, -127.7010956, 127.6975250
24: -79.3418427, 53.0184937, -79.3726883, 53.0300674, -132.3719025, 132.3911743
25: -72.8784790, 62.2197151, -72.9035263, 62.2404366, -135.1189117, 135.1232452
26: -98.6059189, 87.5881500, -98.6279144, 87.6193848, -186.2253113, 186.2160645
27: -85.8276062, 59.0948372, -85.8639526, 59.1111145, -144.9387207, 144.9587708
28: -69.9539413, 63.7523041, -69.9714661, 63.7684746, -133.7224121, 133.7237701
29: -88.5135345, 51.1854858, -88.5469513, 51.2104721, -139.7239990, 139.7324219
30: -87.1540222, 65.4814453, -87.1713715, 65.5129242, -152.6669464, 152.6528015
31: -85.2207870, 55.5220871, -85.2546082, 55.5353699, -140.7561646, 140.7766876
32: -75.8331146, 62.1546898, -75.8515167, 62.1843147, -138.0174103, 138.0061951
33: -108.1132355, 82.5019226, -108.1564331, 82.5177765, -190.6310120, 190.6583557
34: -87.8157349, 65.8787384, -87.8474121, 65.8992462, -153.7149658, 153.7261505
35: -83.9834976, 68.6544952, -84.0205460, 68.6674194, -152.6508942, 152.6750183
36: -82.9299316, 73.6796112, -82.9601059, 73.6932983, -156.6232300, 156.6397095
37: -123.6186905, 71.2336426, -123.6629715, 71.2484131, -194.8670959, 194.8966064
38: -101.3820648, 93.1789780, -101.4180603, 93.2051163, -194.5871735, 194.5970306
39: -115.5723572, 83.5531464, -115.6275024, 83.5659103, -199.1382751, 199.1806335
40: -98.8797226, 60.3155746, -98.9068832, 60.3275032, -159.2072296, 159.2224426
41: -78.8964462, 63.0787048, -78.9230576, 63.0986366, -141.9950867, 142.0017700
42: -63.6430740, 59.0981865, -63.6595154, 59.1481361, -122.7912140, 122.7576981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=448, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=636, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6262029, upper bound: 116.8679766
time: 92.69 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6262029, upper bound: 116.8679766
time: 116.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 211.31 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 211.31
Output dim: 12, lower bound: -116.6262029, upper bound: 116.6363803
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 211.31
Output dim: 12, lower bound: -116.6262029, upper bound: 116.6363803
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 211.31
Output dim: 12, lower bound: -116.6262029, upper bound: 116.8679766
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 211.31
Output dim: 12, lower bound: -116.6262029, upper bound: 116.8679766

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -105.6541290, 76.7902603, -105.6518402, 76.6602707, -182.3143921, 182.4421082
1: -62.6265373, 57.5767288, -62.6017799, 57.4789810, -120.1054993, 120.1785049
2: -58.1855965, 58.0730820, -58.1606026, 57.9456367, -116.1312332, 116.2336807
3: -67.5458832, 66.6710205, -67.4881439, 66.5421371, -134.0880127, 134.1591492
4: -72.8958130, 69.1423035, -72.9038086, 69.0199890, -141.9157867, 142.0461121
5: -69.8522186, 71.5690536, -69.7892227, 71.4214401, -141.2736511, 141.3582764
6: -81.2421570, 61.1121635, -81.2930603, 61.1918716, -142.4340210, 142.4052277
7: -77.5184860, 64.8407745, -77.4810638, 64.7218933, -142.2403870, 142.3218384
8: -81.1717529, 78.5628586, -81.1839218, 78.4372864, -159.6090393, 159.7467804
9: -71.5795441, 68.0438309, -71.5609360, 68.1327362, -139.7122803, 139.6047668
10: -97.5464783, 85.7211685, -97.5032806, 85.8594055, -183.4058838, 183.2244568
11: -92.5011826, 65.0070572, -92.5163345, 65.0963440, -157.5975342, 157.5233917
12: -86.4580002, 88.8853912, -86.3730392, 88.9762650, -175.4342651, 175.2584229
13: -90.1957855, 97.8987045, -90.3160324, 98.0884628, -188.2842407, 188.2147217
14: -133.1210938, 75.5124512, -133.2258606, 75.7409439, -208.8620300, 208.7383118
15: -84.6627274, 63.9170380, -84.7414474, 63.9506264, -148.6133575, 148.6584778
16: -101.3338089, 68.6130676, -101.3361816, 68.7329102, -170.0667114, 169.9492493
17: -135.5495605, 81.2385101, -135.6275330, 81.4164734, -216.9660034, 216.8660278
18: -85.2919769, 68.8337326, -85.3789902, 68.8484192, -154.1403961, 154.2127228
19: -68.3547211, 48.4129982, -68.3339996, 48.3832855, -116.7379837, 116.7469940
20: -60.8190460, 52.6811256, -60.8147736, 52.7043762, -113.5234222, 113.4958801
21: -84.4750519, 57.6003952, -84.4491730, 57.5811882, -142.0562439, 142.0495605
22: -85.1727219, 58.0086098, -85.3330078, 58.0141792, -143.1869049, 143.3416138
23: -69.9051208, 57.3092804, -69.9126740, 57.2980804, -127.2031860, 127.2219543
24: -78.8380814, 52.8463669, -79.0134048, 52.9244232, -131.7624969, 131.8597717
25: -72.5984650, 61.9538918, -72.6643829, 61.9516106, -134.5500793, 134.6182709
26: -98.1709976, 86.9759445, -98.0687714, 86.9037170, -185.0747070, 185.0447083
27: -85.1867065, 58.8654747, -85.3852921, 58.9562454, -144.1429443, 144.2507629
28: -69.6623459, 63.5618744, -69.7122879, 63.6024818, -133.2648315, 133.2741699
29: -88.1960068, 50.8351936, -88.2572708, 50.8230400, -139.0190430, 139.0924683
30: -86.8646774, 65.0741425, -86.8410568, 65.0876160, -151.9523010, 151.9151917
31: -84.7898788, 55.3824997, -84.8371506, 55.3682327, -140.1581116, 140.2196350
32: -75.5360565, 61.7037659, -75.5409622, 61.7891121, -137.3251648, 137.2447205
33: -107.4236755, 82.1329193, -107.5968323, 82.1832733, -189.6069489, 189.7297516
34: -87.3662643, 65.5425720, -87.4601288, 65.6183929, -152.9846497, 153.0027008
35: -83.4403763, 68.3345642, -83.5738678, 68.3887558, -151.8291321, 151.9084320
36: -82.5790863, 73.4818726, -82.6768494, 73.5072479, -156.0863190, 156.1587219
37: -123.0450134, 70.9729309, -123.2418976, 70.9862518, -194.0312500, 194.2148285
38: -100.9295959, 92.9313507, -100.9850311, 92.9153442, -193.8449402, 193.9163666
39: -115.1295090, 83.3595886, -115.2375107, 83.3544159, -198.4839172, 198.5971069
40: -98.3250122, 60.0777359, -98.4964752, 60.1478157, -158.4728088, 158.5742188
41: -78.4830399, 62.8665428, -78.6087952, 62.8895493, -141.3725586, 141.4753418
42: -63.3317604, 58.5965538, -63.3351555, 58.6381607, -121.9699173, 121.9317093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=635, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1315

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6020471, upper bound: 116.4072846
time: 100.62 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6020471, upper bound: 116.6138088
time: 86.51 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -105.7384338, 76.8086090, -106.0486221, 77.0106049, -182.7490387, 182.8572235
1: -62.6833229, 57.5888252, -62.8641815, 57.7177849, -120.4011078, 120.4530029
2: -58.2851562, 58.0853462, -58.5838013, 58.2458382, -116.5309906, 116.6691437
3: -67.6506805, 66.6907806, -67.9319611, 66.8986053, -134.5492859, 134.6227417
4: -73.0085754, 69.1587372, -73.3850250, 69.3614426, -142.3700104, 142.5437622
5: -69.9612427, 71.5887604, -70.2552185, 71.8243713, -141.7856140, 141.8439789
6: -81.2755814, 61.1435699, -81.5325623, 61.3752174, -142.6507874, 142.6761322
7: -77.6038361, 64.8555298, -77.8944168, 65.0183411, -142.6221619, 142.7499390
8: -81.2602005, 78.5811386, -81.5741272, 78.7789764, -160.0391846, 160.1552734
9: -71.6053009, 68.1261139, -71.8376007, 68.5060272, -140.1113281, 139.9637146
10: -97.5880966, 85.8926239, -98.0873489, 86.5764999, -184.1645966, 183.9799805
11: -92.5282288, 65.1238098, -92.9549942, 65.5775452, -158.1057739, 158.0787964
12: -86.4793701, 89.0712433, -86.9896698, 89.7443085, -176.2236786, 176.0609131
13: -90.2210083, 97.9494324, -90.4761963, 98.3899994, -188.6109924, 188.4256287
14: -133.1677399, 75.6377945, -133.7575836, 76.2618866, -209.4296265, 209.3953857
15: -84.7580719, 63.9446487, -85.1743011, 64.2229919, -148.9810486, 149.1189270
16: -101.3740158, 68.6998672, -101.7066650, 69.1230164, -170.4970245, 170.4065247
17: -135.5773468, 81.3373260, -136.0749207, 81.8715668, -217.4488831, 217.4122467
18: -85.3432846, 68.8746338, -85.7396088, 69.0595398, -154.4028320, 154.6142426
19: -68.3816376, 48.4475784, -68.6131439, 48.5419769, -116.9236145, 117.0607224
20: -60.8447571, 52.7263870, -61.0720940, 52.9018974, -113.7466431, 113.7984772
21: -84.5018692, 57.6709404, -84.8222504, 57.8859062, -142.3877716, 142.4931946
22: -85.2023621, 58.0591774, -85.5560760, 58.2853813, -143.4877319, 143.6152496
23: -69.9286957, 57.3432350, -70.1438065, 57.4709244, -127.3996124, 127.4870377
24: -78.8800278, 52.8589058, -79.2404327, 53.0051956, -131.8852234, 132.0993347
25: -72.6200485, 62.0003815, -72.8368988, 62.1804619, -134.8005066, 134.8372803
26: -98.2025223, 87.1053925, -98.5578156, 87.4683685, -185.6708679, 185.6631927
27: -85.2475204, 58.8806190, -85.7014771, 59.0815392, -144.3290558, 144.5820923
28: -69.6855774, 63.5831413, -69.9051056, 63.7249184, -133.4104919, 133.4882507
29: -88.2183380, 50.9022789, -88.4781265, 51.1221809, -139.3405151, 139.3804016
30: -86.8865891, 65.1467133, -87.1121902, 65.4069443, -152.2935333, 152.2588806
31: -84.8297806, 55.4087219, -85.1632996, 55.5029564, -140.3327332, 140.5720062
32: -75.5612793, 61.7676468, -75.7925339, 62.0692711, -137.6305542, 137.5601807
33: -107.5064392, 82.1594086, -107.9764099, 82.4735413, -189.9799805, 190.1358185
34: -87.4207840, 65.5663605, -87.7330322, 65.8571472, -153.2779236, 153.2993927
35: -83.5081329, 68.3554535, -83.8752594, 68.6307678, -152.1389008, 152.2307129
36: -82.6125259, 73.5032120, -82.8694458, 73.6562576, -156.2687836, 156.3726501
37: -123.0860596, 71.0101624, -123.5205078, 71.1923447, -194.2784119, 194.5306702
38: -100.9901199, 92.9515076, -101.3104935, 93.1598587, -194.1499634, 194.2619934
39: -115.1733780, 83.3801575, -115.5229721, 83.5327530, -198.7061157, 198.9031067
40: -98.3686829, 60.0901375, -98.7769318, 60.3008957, -158.6695862, 158.8670654
41: -78.5192261, 62.8908195, -78.8228989, 63.0407104, -141.5599365, 141.7137146
42: -63.3560715, 58.6730576, -63.6048241, 59.0113792, -122.3674393, 122.2778778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=635, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8413660, upper bound: 116.4072846
time: 94.71 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8413660, upper bound: 116.6138088
time: 544.98 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -106.0402832, 77.0226364, -105.7348328, 76.6971436, -182.7374268, 182.7574768
1: -62.8494911, 57.7202110, -62.6476021, 57.5017319, -120.3512268, 120.3678131
2: -58.5728683, 58.2490501, -58.2550430, 57.9678307, -116.5406952, 116.5040894
3: -67.9035950, 66.9078140, -67.5726471, 66.5785370, -134.4821320, 134.4804688
4: -73.3903275, 69.3694534, -73.0322723, 69.0528412, -142.4431763, 142.4017334
5: -70.2195663, 71.8344116, -69.8726120, 71.4587479, -141.6782990, 141.7070312
6: -81.5431366, 61.3962440, -81.3485413, 61.2565460, -142.7996826, 142.7447815
7: -77.8665466, 65.0226440, -77.5474854, 64.7535248, -142.6200714, 142.5701294
8: -81.5707855, 78.7904892, -81.2778778, 78.4738617, -160.0446472, 160.0683594
9: -71.8364716, 68.5358124, -71.5936661, 68.2569427, -140.0933838, 140.1294861
10: -98.0909271, 86.6239777, -97.5596695, 86.0961914, -184.1870880, 184.1836395
11: -92.9663849, 65.5993652, -92.5625000, 65.2503738, -158.2167511, 158.1618652
12: -86.9976959, 89.7737045, -86.4130859, 89.2051163, -176.2028198, 176.1867828
13: -90.4725723, 98.4532928, -90.3494110, 98.2161026, -188.6886749, 188.8027039
14: -133.7578430, 76.3344116, -133.2889404, 75.9589691, -209.7168121, 209.6233521
15: -85.1914291, 64.2339783, -84.8689499, 63.9932289, -149.1846619, 149.1029205
16: -101.7143402, 69.1542435, -101.3954468, 68.8680878, -170.5824127, 170.5496826
17: -136.0724487, 81.9332886, -135.6676331, 81.5918274, -217.6642456, 217.6009216
18: -85.7675858, 69.0477982, -85.4766998, 68.8878555, -154.6554413, 154.5244751
19: -68.6026154, 48.5225906, -68.3793793, 48.4038544, -117.0064697, 116.9019699
20: -61.0801201, 52.9044456, -60.8529434, 52.7592659, -113.8393860, 113.7573853
21: -84.8340683, 57.8765335, -84.4970245, 57.6465874, -142.4806519, 142.3735504
22: -85.5719833, 58.2843323, -85.4189224, 58.0621185, -143.6340942, 143.7032471
23: -70.1527252, 57.4568405, -69.9563141, 57.3280716, -127.4807892, 127.4131393
24: -79.2931061, 53.0034447, -79.1227264, 52.9417953, -132.2348938, 132.1261597
25: -72.8531799, 62.1682243, -72.7202225, 61.9945068, -134.8476868, 134.8884277
26: -98.5697174, 87.4423370, -98.1244888, 87.0120697, -185.5817719, 185.5668335
27: -85.7596054, 59.0767975, -85.5225220, 58.9760933, -144.7357025, 144.5993195
28: -69.9275818, 63.7260094, -69.7690277, 63.6286125, -133.5561829, 133.4950409
29: -88.4863663, 51.1098518, -88.3135376, 50.8831482, -139.3695068, 139.4233856
30: -87.1281738, 65.4013214, -86.8883591, 65.1676025, -152.2957764, 152.2896729
31: -85.1755600, 55.4921112, -84.9117203, 55.3879700, -140.5635223, 140.4038391
32: -75.8040390, 62.0846176, -75.5870972, 61.8855362, -137.6895752, 137.6717072
33: -108.0213165, 82.4722748, -107.7451019, 82.2161407, -190.2374573, 190.2173767
34: -87.7550049, 65.8515854, -87.5529709, 65.6488876, -153.4039001, 153.4045563
35: -83.9076920, 68.6316681, -83.6913223, 68.4175873, -152.3252869, 152.3229980
36: -82.8899002, 73.6557312, -82.7452927, 73.5338364, -156.4237213, 156.4010162
37: -123.5700760, 71.1907349, -123.3611908, 71.0192261, -194.5892944, 194.5519257
38: -101.3137512, 93.1557388, -101.0655975, 92.9510269, -194.2647705, 194.2213440
39: -115.5218811, 83.5299530, -115.3202591, 83.3781891, -198.9000397, 198.8501892
40: -98.8294601, 60.3005791, -98.6049957, 60.1657944, -158.9952393, 158.9055786
41: -78.8552551, 63.0483170, -78.6932602, 62.9231224, -141.7783813, 141.7415771
42: -63.6151657, 59.0049553, -63.3789520, 58.7397957, -122.3549652, 122.3839111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=636, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6020471, upper bound: 116.6390472
time: 126.72 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6020471, upper bound: 116.8413658
time: 86.02 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -106.1248932, 77.0412750, -106.1332626, 77.0475082, -183.1723938, 183.1745300
1: -62.9061546, 57.7326241, -62.9099960, 57.7407036, -120.6468582, 120.6426239
2: -58.6759300, 58.2614021, -58.6813393, 58.2681541, -116.9440842, 116.9427338
3: -68.0114746, 66.9279327, -68.0197906, 66.9352875, -134.9467621, 134.9477234
4: -73.5031738, 69.3862457, -73.5138855, 69.3945770, -142.8977509, 142.9001312
5: -70.3299255, 71.8541870, -70.3428192, 71.8618774, -142.1918030, 142.1970062
6: -81.5769730, 61.4281731, -81.5885468, 61.4400177, -143.0169678, 143.0167236
7: -77.9524231, 65.0379410, -77.9631653, 65.0500183, -143.0024414, 143.0010986
8: -81.6591339, 78.8088913, -81.6689301, 78.8156891, -160.4748230, 160.4778137
9: -71.8623581, 68.6181870, -71.8707962, 68.6304932, -140.4928589, 140.4889832
10: -98.1324005, 86.7947845, -98.1437149, 86.8135757, -184.9459839, 184.9385071
11: -92.9934006, 65.7165070, -93.0009537, 65.7325287, -158.7259216, 158.7174683
12: -87.0193634, 89.9592972, -87.0301132, 89.9732437, -176.9926147, 176.9893799
13: -90.4985199, 98.5050354, -90.5105972, 98.5183411, -189.0168610, 189.0156250
14: -133.8048859, 76.4643173, -133.8206787, 76.4802856, -210.2851715, 210.2850037
15: -85.2876205, 64.2613678, -85.3034668, 64.2655792, -149.5531769, 149.5648346
16: -101.7549057, 69.2414856, -101.7664948, 69.2596436, -171.0145264, 171.0079651
17: -136.1007996, 82.0357895, -136.1154022, 82.0479889, -218.1487732, 218.1511841
18: -85.8197861, 69.0890198, -85.8380127, 69.1000977, -154.9198761, 154.9270325
19: -68.6299667, 48.5576324, -68.6568451, 48.5629883, -117.1929321, 117.2144775
20: -61.1059685, 52.9498940, -61.1103210, 52.9571915, -114.0631409, 114.0602112
21: -84.8609314, 57.9468994, -84.8702011, 57.9515800, -142.8125000, 142.8170929
22: -85.6026230, 58.3351288, -85.6427994, 58.3336182, -143.9362488, 143.9779358
23: -70.1767120, 57.4946404, -70.1874619, 57.5035934, -127.6802979, 127.6820984
24: -79.3362503, 53.0164413, -79.3509598, 53.0232162, -132.3594360, 132.3674011
25: -72.8753738, 62.2149658, -72.8922119, 62.2246742, -135.1000519, 135.1071777
26: -98.6019363, 87.5780029, -98.6138992, 87.5824738, -186.1844177, 186.1918945
27: -85.8209076, 59.0921211, -85.8391800, 59.1017876, -144.9226990, 144.9313049
28: -69.9513092, 63.7475700, -69.9620132, 63.7517624, -133.7030640, 133.7095795
29: -88.5096130, 51.1771545, -88.5330811, 51.1832466, -139.6928558, 139.7102356
30: -87.1506195, 65.4743652, -87.1591492, 65.4877625, -152.6383820, 152.6335144
31: -85.2162628, 55.5186043, -85.2382965, 55.5233727, -140.7396240, 140.7568970
32: -75.8295517, 62.1510124, -75.8388367, 62.1674919, -137.9970398, 137.9898529
33: -108.1047974, 82.4987030, -108.1255951, 82.5064697, -190.6112671, 190.6242981
34: -87.8098984, 65.8754578, -87.8263474, 65.8877716, -153.6976624, 153.7017822
35: -83.9760361, 68.6522369, -83.9934006, 68.6592102, -152.6352386, 152.6456299
36: -82.9239807, 73.6767731, -82.9386902, 73.6831360, -156.6071167, 156.6154633
37: -123.6125336, 71.2274323, -123.6411133, 71.2261276, -194.8386536, 194.8685455
38: -101.3745422, 93.1761780, -101.3919449, 93.1949768, -194.5695190, 194.5681152
39: -115.5664825, 83.5504913, -115.6065140, 83.5565948, -199.1230774, 199.1570129
40: -98.8740540, 60.3131676, -98.8866119, 60.3188782, -159.1929169, 159.1997833
41: -78.8919525, 63.0717735, -78.9073029, 63.0744400, -141.9664001, 141.9790802
42: -63.6396599, 59.0874443, -63.6478462, 59.1140976, -122.7537537, 122.7352905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=636, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8413660, upper bound: 116.6390472
time: 99.51 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6020471, upper bound: 116.8413658
time: 107.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 209.26 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 209.26
Output dim: 12, lower bound: -116.6020471, upper bound: 116.4072846
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 209.26
Output dim: 12, lower bound: -116.6020471, upper bound: 116.6138088
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 209.26
Output dim: 12, lower bound: -116.8413660, upper bound: 116.4072846
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 209.26
Output dim: 12, lower bound: -116.8413660, upper bound: 116.6138088
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 209.26
Output dim: 12, lower bound: -116.6020471, upper bound: 116.6390472
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 209.26
Output dim: 12, lower bound: -116.6020471, upper bound: 116.8413658
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 209.26
Output dim: 12, lower bound: -116.8413660, upper bound: 116.6390472
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 209.26
Output dim: 12, lower bound: -116.6020471, upper bound: 116.8413658

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -105.2675095, 76.5942993, -105.5564270, 76.6298676, -181.8973694, 182.1507263
1: -62.3921852, 57.4590797, -62.5442924, 57.4582901, -119.8504715, 120.0033722
2: -57.7748528, 57.9040794, -58.0504303, 57.9262314, -115.7010803, 115.9545059
3: -67.1360703, 66.4588776, -67.3789215, 66.5096588, -133.6457214, 133.8377991
4: -72.4171600, 68.9581604, -72.7744980, 68.9926758, -141.4098206, 141.7326355
5: -69.4365234, 71.2991867, -69.6764526, 71.3873672, -140.8238831, 140.9756470
6: -81.0160751, 60.7856865, -81.2491684, 61.1068039, -142.1228790, 142.0348511
7: -77.1378098, 64.6862564, -77.3875504, 64.6947174, -141.8325195, 142.0737915
8: -80.7633209, 78.3746185, -81.0762405, 78.4069672, -159.1702881, 159.4508667
9: -71.4271469, 67.6059113, -71.5271149, 68.0229874, -139.4501343, 139.1330261
10: -97.2931595, 85.0050278, -97.4533539, 85.6724701, -182.9656219, 182.4583740
11: -92.1837234, 64.4873810, -92.4717255, 64.9561920, -157.1399231, 156.9591064
12: -86.0061188, 87.9467316, -86.3367538, 88.7218781, -174.7279816, 174.2834625
13: -89.9522400, 97.3513718, -90.2800522, 97.9515228, -187.9037628, 187.6314087
14: -132.6231384, 74.7957306, -133.1578979, 75.5457916, -208.1689301, 207.9536133
15: -84.1657104, 63.6656189, -84.6102905, 63.9131203, -148.0788269, 148.2758942
16: -100.9701385, 68.1726685, -101.2606812, 68.6214218, -169.5915222, 169.4333496
17: -135.0942688, 80.5443268, -135.5812378, 81.2302094, -216.3244629, 216.1255646
18: -84.8739243, 68.5995789, -85.2810898, 68.7934570, -153.6673889, 153.8806610
19: -68.1219940, 48.2752075, -68.2874603, 48.3491135, -116.4710999, 116.5626526
20: -60.6107178, 52.4433174, -60.7780113, 52.6436195, -113.2543335, 113.2213135
21: -84.2068405, 57.3097916, -84.4052734, 57.5072479, -141.7140808, 141.7150574
22: -84.8773270, 57.7053299, -85.2659149, 57.9455986, -142.8229065, 142.9712524
23: -69.6657410, 57.1680222, -69.8638763, 57.2654266, -126.9311523, 127.0318985
24: -78.4304199, 52.7272644, -78.9119720, 52.9059486, -131.3363647, 131.6392212
25: -72.3378448, 61.7321434, -72.6034927, 61.8995056, -134.2373352, 134.3356323
26: -97.8291855, 86.4503632, -98.0163498, 86.7652130, -184.5943909, 184.4667053
27: -84.7167130, 58.7432137, -85.2673111, 58.9372559, -143.6539612, 144.0105286
28: -69.4410477, 63.4530220, -69.6652527, 63.5764961, -133.0175171, 133.1182709
29: -87.9659576, 50.4979973, -88.2099380, 50.7352791, -138.7012329, 138.7079163
30: -86.6272659, 64.7902985, -86.7926025, 65.0134888, -151.6407471, 151.5829010
31: -84.4137115, 55.2691650, -84.7538071, 55.3429337, -139.7566528, 140.0229797
32: -75.3088226, 61.2760048, -75.5029144, 61.6743927, -136.9832153, 136.7789001
33: -106.8737717, 81.9341278, -107.4617004, 82.1472931, -189.0210571, 189.3958282
34: -87.0647354, 65.3585663, -87.3880463, 65.5882721, -152.6529999, 152.7466125
35: -83.0760727, 68.1837311, -83.4854202, 68.3603210, -151.4363861, 151.6691284
36: -82.3782043, 73.2466736, -82.6403656, 73.4487839, -155.8269958, 155.8870392
37: -122.6382065, 70.7709656, -123.1522980, 70.9385834, -193.5767822, 193.9232635
38: -100.5934372, 92.6216278, -100.9221420, 92.8393097, -193.4327393, 193.5437622
39: -114.7743073, 83.1509399, -115.1641922, 83.3094635, -198.0837708, 198.3151093
40: -97.9050903, 59.9371948, -98.3989182, 60.1274261, -158.0325012, 158.3361053
41: -78.2186813, 62.6557236, -78.5494995, 62.8389778, -141.0576477, 141.2052155
42: -63.1298103, 58.1809807, -63.3011971, 58.5282936, -121.6581039, 121.4821625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=446, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=634, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.4306273, upper bound: 116.3779140
time: 78.73 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.4306273, upper bound: 116.3779140
time: 77.68 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -105.6253357, 76.7814941, -105.6430130, 76.6580048, -182.2833405, 182.4244995
1: -62.6061211, 57.5674095, -62.5957565, 57.4765701, -120.0826874, 120.1631622
2: -58.1551437, 58.0660706, -58.1528816, 57.9438324, -116.0989685, 116.2189484
3: -67.5204010, 66.6605835, -67.4810333, 66.5395660, -134.0599670, 134.1416168
4: -72.8593750, 69.1313705, -72.8949127, 69.0171509, -141.8765106, 142.0262756
5: -69.8252716, 71.5591583, -69.7810059, 71.4189224, -141.2442017, 141.3401642
6: -81.2277985, 61.0827293, -81.2891846, 61.1850510, -142.4128418, 142.3719177
7: -77.4905243, 64.8308868, -77.4729080, 64.7192841, -142.2097931, 142.3037872
8: -81.1403198, 78.5527191, -81.1759033, 78.4346619, -159.5749512, 159.7286224
9: -71.5694275, 68.0121918, -71.5578079, 68.1252747, -139.6947021, 139.5700073
10: -97.5289764, 85.6717148, -97.4987564, 85.8475800, -183.3765564, 183.1704559
11: -92.4858475, 64.9680328, -92.5122986, 65.0863647, -157.5722046, 157.4803314
12: -86.4456100, 88.8232574, -86.3697662, 88.9610977, -175.4066925, 175.1930084
13: -90.1813812, 97.8622360, -90.3117294, 98.0796509, -188.2610321, 188.1739655
14: -133.0956421, 75.4686737, -133.2193298, 75.7305756, -208.8262177, 208.6880035
15: -84.6219788, 63.9046555, -84.7310028, 63.9475937, -148.5695801, 148.6356506
16: -101.2989807, 68.5764465, -101.3271561, 68.7240753, -170.0230408, 169.9035950
17: -135.5326233, 81.1938477, -135.6228943, 81.4055023, -216.9381104, 216.8167419
18: -85.2581863, 68.8095703, -85.3708191, 68.8420181, -154.1002045, 154.1803894
19: -68.3253479, 48.4022255, -68.3265228, 48.3804970, -116.7058411, 116.7287445
20: -60.8088188, 52.6623764, -60.8120804, 52.6996613, -113.5084686, 113.4744568
21: -84.4605560, 57.5791473, -84.4454346, 57.5757217, -142.0362854, 142.0245819
22: -85.1347809, 57.9822197, -85.3235550, 58.0072479, -143.1420288, 143.3057709
23: -69.8811798, 57.2930298, -69.9067535, 57.2938766, -127.1750488, 127.1997833
24: -78.8062439, 52.8401947, -79.0058212, 52.9217758, -131.7280273, 131.8459930
25: -72.5738907, 61.9401665, -72.6582642, 61.9475822, -134.5214691, 134.5984344
26: -98.1537628, 86.9399719, -98.0643005, 86.8942795, -185.0480194, 185.0042419
27: -85.1528397, 58.8557396, -85.3770523, 58.9534187, -144.1062622, 144.2327881
28: -69.6416321, 63.5486298, -69.7075806, 63.5990906, -133.2407227, 133.2562103
29: -88.1704254, 50.8179817, -88.2505417, 50.8181076, -138.9885254, 139.0685120
30: -86.8382721, 65.0440369, -86.8345032, 65.0787201, -151.9169922, 151.8785400
31: -84.7629929, 55.3701134, -84.8301849, 55.3647308, -140.1277161, 140.2002869
32: -75.5241547, 61.6708755, -75.5375061, 61.7811584, -137.3053131, 137.2083740
33: -107.3870773, 82.1217041, -107.5878143, 82.1789856, -189.5660706, 189.7094879
34: -87.3440323, 65.5306168, -87.4545441, 65.6139984, -152.9580231, 152.9851532
35: -83.4129333, 68.3180847, -83.5671158, 68.3838959, -151.7968292, 151.8851929
36: -82.5672684, 73.4658051, -82.6735840, 73.5030823, -156.0703278, 156.1393890
37: -123.0043259, 70.9530487, -123.2323227, 70.9807663, -193.9850922, 194.1853638
38: -100.9082794, 92.9112320, -100.9793701, 92.9109955, -193.8192749, 193.8905945
39: -115.1022873, 83.3439484, -115.2306061, 83.3492813, -198.4515686, 198.5745392
40: -98.3017731, 60.0677986, -98.4904785, 60.1445160, -158.4462891, 158.5582581
41: -78.4584122, 62.8485146, -78.6023254, 62.8850708, -141.3434753, 141.4508362
42: -63.3216553, 58.5578346, -63.3321800, 58.6284828, -121.9501266, 121.8900070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=446, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=635, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.4306273, upper bound: 116.5863312
time: 107.27 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.4306273, upper bound: 116.5863312
time: 132.16 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -105.3518906, 76.6127319, -105.9530258, 76.9803925, -182.3322754, 182.5657654
1: -62.4490356, 57.4711342, -62.8067169, 57.6971970, -120.1462326, 120.2778473
2: -57.8745461, 57.9164810, -58.4736900, 58.2265244, -116.1010742, 116.3901672
3: -67.2409515, 66.4787140, -67.8227386, 66.8663940, -134.1073456, 134.3014526
4: -72.5300598, 68.9746857, -73.2557755, 69.3342743, -141.8643341, 142.2304688
5: -69.5455475, 71.3191223, -70.1424026, 71.7905426, -141.3360901, 141.4615173
6: -81.0495834, 60.8159218, -81.4886856, 61.2902069, -142.3397827, 142.3045959
7: -77.2231369, 64.7010040, -77.8004456, 64.9912643, -142.2144012, 142.5014343
8: -80.8517838, 78.3928680, -81.4664001, 78.7488403, -159.6006165, 159.8592529
9: -71.4532166, 67.6881790, -71.8041458, 68.3962555, -139.8494720, 139.4923248
10: -97.3349991, 85.1766739, -98.0377197, 86.3895721, -183.7245789, 183.2143707
11: -92.2110901, 64.6043091, -92.9108047, 65.4374008, -157.6484985, 157.5151062
12: -86.0276794, 88.1325455, -86.9535294, 89.4898529, -175.5175171, 175.0860596
13: -89.9774933, 97.4016495, -90.4402771, 98.2529831, -188.2304535, 187.8419189
14: -132.6699982, 74.9209518, -133.6899414, 76.0666046, -208.7366028, 208.6108704
15: -84.2610931, 63.6933327, -85.0430908, 64.1856842, -148.4467773, 148.7364197
16: -101.0101089, 68.2589874, -101.6321487, 69.0110703, -170.0211792, 169.8911438
17: -135.1222229, 80.6429214, -136.0287781, 81.6851196, -216.8073425, 216.6716919
18: -84.9246902, 68.6404877, -85.6414032, 69.0044556, -153.9291382, 154.2818756
19: -68.1486206, 48.3099785, -68.5668182, 48.5077591, -116.6563797, 116.8768005
20: -60.6365891, 52.4886513, -61.0356445, 52.8411255, -113.4777069, 113.5242920
21: -84.2338257, 57.3804665, -84.7786102, 57.8119545, -142.0457764, 142.1590729
22: -84.9067993, 57.7555885, -85.4890900, 58.2167244, -143.1235199, 143.2446747
23: -69.6890106, 57.2020760, -70.0952835, 57.4381104, -127.1271210, 127.2973480
24: -78.4719467, 52.7398720, -79.1382446, 52.9868050, -131.4587555, 131.8781128
25: -72.3592911, 61.7785492, -72.7760696, 62.1282578, -134.4875488, 134.5546112
26: -97.8608856, 86.5799255, -98.5058594, 87.3293457, -185.1902313, 185.0857849
27: -84.7770844, 58.7584763, -85.5832443, 59.0625725, -143.8396606, 144.3417206
28: -69.4640503, 63.4743690, -69.8590088, 63.6989784, -133.1630249, 133.3333740
29: -87.9882431, 50.5648956, -88.4310074, 51.0344353, -139.0226746, 138.9958954
30: -86.6490097, 64.8627090, -87.0640717, 65.3327789, -151.9817810, 151.9267731
31: -84.4530640, 55.2954330, -85.0800552, 55.4776268, -139.9306793, 140.3754883
32: -75.3342133, 61.3398895, -75.7548904, 61.9545326, -137.2887421, 137.0947571
33: -106.9564972, 81.9607849, -107.8412399, 82.4377899, -189.3942871, 189.8020325
34: -87.1191177, 65.3825226, -87.6607590, 65.8273010, -152.9464111, 153.0432739
35: -83.1437988, 68.2045746, -83.7866516, 68.6026764, -151.7464752, 151.9912262
36: -82.4115753, 73.2678680, -82.8329010, 73.5984268, -156.0099945, 156.1007385
37: -122.6789398, 70.8081207, -123.4307861, 71.1443787, -193.8233185, 194.2389069
38: -100.6541977, 92.6415253, -101.2473984, 93.0856018, -193.7397919, 193.8889160
39: -114.8179703, 83.1714249, -115.4499817, 83.4880524, -198.3060150, 198.6213989
40: -97.9486084, 59.9496231, -98.6792984, 60.2805367, -158.2291260, 158.6289215
41: -78.2548141, 62.6801033, -78.7633896, 62.9900436, -141.2448578, 141.4434967
42: -63.1542816, 58.2570877, -63.5716248, 58.9009056, -122.0551910, 121.8287125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=446, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=634, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.4306273, upper bound: 116.3779140
time: 106.37 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.4306273, upper bound: 116.3779140
time: 82.62 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -105.7096252, 76.7997665, -106.0397949, 77.0083160, -182.7179413, 182.8395691
1: -62.6629486, 57.5793877, -62.8582115, 57.7153206, -120.3782654, 120.4375992
2: -58.2547150, 58.0783119, -58.5761337, 58.2440033, -116.4987183, 116.6544418
3: -67.6252060, 66.6803055, -67.9249649, 66.8959656, -134.5211639, 134.6052551
4: -72.9721451, 69.1478577, -73.3761597, 69.3585968, -142.3307495, 142.5240173
5: -69.9342957, 71.5788879, -70.2470322, 71.8217926, -141.7560883, 141.8259277
6: -81.2612305, 61.1142654, -81.5286560, 61.3683205, -142.6295471, 142.6428986
7: -77.5743103, 64.8455734, -77.8854065, 65.0156250, -142.5899353, 142.7309723
8: -81.2287445, 78.5708923, -81.5661774, 78.7763367, -160.0050659, 160.1370697
9: -71.5951157, 68.0945206, -71.8343124, 68.4986496, -140.0937653, 139.9288330
10: -97.5705872, 85.8431396, -98.0828094, 86.5647125, -184.1352997, 183.9259338
11: -92.5129395, 65.0847015, -92.9510193, 65.5674744, -158.0804138, 158.0357056
12: -86.4669495, 89.0090942, -86.9863281, 89.7291260, -176.1960754, 175.9954224
13: -90.2064590, 97.9130096, -90.4717636, 98.3811111, -188.5875702, 188.3847656
14: -133.1422424, 75.5939941, -133.7509766, 76.2514801, -209.3937225, 209.3449707
15: -84.7170029, 63.9322929, -85.1636200, 64.2199554, -148.9369507, 149.0959167
16: -101.3391266, 68.6630859, -101.6975632, 69.1140594, -170.4531860, 170.3606567
17: -135.5604248, 81.2926483, -136.0702820, 81.8605499, -217.4209595, 217.3629150
18: -85.3094406, 68.8503647, -85.7314224, 69.0529938, -154.3624268, 154.5817871
19: -68.3523560, 48.4368248, -68.6057358, 48.5391922, -116.8915405, 117.0425568
20: -60.8344917, 52.7076378, -61.0693779, 52.8971176, -113.7315979, 113.7770081
21: -84.4874344, 57.6496658, -84.8185730, 57.8804169, -142.3678589, 142.4682312
22: -85.1642609, 58.0328102, -85.5465012, 58.2784882, -143.4427338, 143.5793152
23: -69.9046936, 57.3269691, -70.1378860, 57.4666443, -127.3713379, 127.4648438
24: -78.8482056, 52.8526535, -79.2328033, 53.0024567, -131.8506622, 132.0854492
25: -72.5956192, 61.9865723, -72.8307800, 62.1763306, -134.7719421, 134.8173523
26: -98.1851730, 87.0694427, -98.5532761, 87.4588623, -185.6440125, 185.6226959
27: -85.2136307, 58.8708725, -85.6931763, 59.0786591, -144.2922974, 144.5640564
28: -69.6648102, 63.5698509, -69.9003906, 63.7215233, -133.3863220, 133.4702301
29: -88.1925964, 50.8836708, -88.4712448, 51.1164055, -139.3090057, 139.3549194
30: -86.8601303, 65.1165543, -87.1056061, 65.3980103, -152.2581177, 152.2221680
31: -84.8029251, 55.3962212, -85.1563187, 55.4993668, -140.3022766, 140.5525360
32: -75.5493546, 61.7347336, -75.7890167, 62.0613441, -137.6107025, 137.5237427
33: -107.4698334, 82.1480637, -107.9673233, 82.4692001, -189.9390259, 190.1153717
34: -87.3985443, 65.5543213, -87.7273483, 65.8526764, -153.2512207, 153.2816772
35: -83.4806519, 68.3388596, -83.8683853, 68.6259155, -152.1065674, 152.2072449
36: -82.6005402, 73.4870987, -82.8660355, 73.6521149, -156.2526550, 156.3531342
37: -123.0454407, 70.9901962, -123.5108185, 71.1868591, -194.2322693, 194.5010071
38: -100.9687347, 92.9312897, -101.3047714, 93.1546097, -194.1233215, 194.2360535
39: -115.1460800, 83.3644257, -115.5159912, 83.5276794, -198.6737671, 198.8804016
40: -98.3454971, 60.0801430, -98.7707825, 60.2975693, -158.6430511, 158.8509216
41: -78.4946747, 62.8729744, -78.8164673, 63.0362549, -141.5309296, 141.6894379
42: -63.3459702, 58.6343613, -63.6018867, 59.0016441, -122.3476105, 122.2362366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=446, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=635, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.4306273, upper bound: 116.5863312
time: 1793.49 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.4306273, upper bound: 116.5863312
time: 78.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 1874.59 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1874.59
Output dim: 12, lower bound: -116.4306273, upper bound: 116.3779140
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1874.59
Output dim: 12, lower bound: -116.4306273, upper bound: 116.3779140
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1874.59
Output dim: 12, lower bound: -116.4306273, upper bound: 116.5863312
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1874.59
Output dim: 12, lower bound: -116.4306273, upper bound: 116.5863312
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1874.59
Output dim: 12, lower bound: -116.4306273, upper bound: 116.3779140
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1874.59
Output dim: 12, lower bound: -116.4306273, upper bound: 116.3779140
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1874.59
Output dim: 12, lower bound: -116.4306273, upper bound: 116.5863312
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1874.59
Output dim: 12, lower bound: -116.4306273, upper bound: 116.5863312
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1874.59
Output dim: 12, lower bound: -116.6020471, upper bound: 116.6390472
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1874.59
Output dim: 12, lower bound: -116.6020471, upper bound: 116.8413658
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1874.59
Output dim: 12, lower bound: -116.8413660, upper bound: 116.6390472
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1874.59
Output dim: 12, lower bound: -116.6020471, upper bound: 116.8413658
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=177.0996856689453
rel_dist={12: [-116.89836616746769, 116.89836617699747]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

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
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8894997, upper bound: 112.7313450
time: 1171.54 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8894997, upper bound: 112.8894995
time: 97.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1268.91 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1268.91
Output dim: 12, lower bound: -112.8894997, upper bound: 112.7313450
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1268.91
Output dim: 12, lower bound: -112.8894997, upper bound: 112.8894995

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -105.7478790, 76.8107834, -106.0065231, 76.9849396, -182.7328033, 182.8173065
1: -62.6895599, 57.5903893, -62.8450584, 57.7012596, -120.3908234, 120.4354477
2: -58.2942429, 58.0871048, -58.5300217, 58.2307625, -116.5250092, 116.6171188
3: -67.6614227, 66.6938324, -67.8886566, 66.8755035, -134.5369263, 134.5824890
4: -73.0185394, 69.1616516, -73.3102264, 69.3398132, -142.3583527, 142.4718781
5: -69.9693909, 71.5915070, -70.2037506, 71.7996979, -141.7690735, 141.7952576
6: -81.2805939, 61.1545334, -81.4976120, 61.3520355, -142.6326294, 142.6521301
7: -77.6139984, 64.8577576, -77.8635635, 64.9953918, -142.6093903, 142.7213135
8: -81.2684784, 78.5836639, -81.5197754, 78.7545853, -160.0230713, 160.1034241
9: -71.6087952, 68.1343994, -71.8182068, 68.4256287, -140.0344238, 139.9526062
10: -97.5931625, 85.9088516, -98.0527420, 86.4266663, -184.0198364, 183.9615784
11: -92.5323410, 65.1348572, -92.9269638, 65.4797058, -158.0120392, 158.0618134
12: -86.4826508, 89.0872192, -86.9627075, 89.6017609, -176.0843964, 176.0499268
13: -90.2257690, 97.9576340, -90.4580231, 98.3059158, -188.5316467, 188.4156494
14: -133.1739807, 75.6484756, -133.7192078, 76.1101074, -209.2840881, 209.3676758
15: -84.7683563, 63.9488258, -85.0948334, 64.1985626, -148.9669189, 149.0436554
16: -101.3795853, 68.7092209, -101.6710281, 69.0344009, -170.4139862, 170.3802490
17: -135.5813141, 81.3465424, -136.0486755, 81.7477646, -217.3290710, 217.3952179
18: -85.3504333, 68.8810425, -85.6758957, 69.0420990, -154.3925171, 154.5569305
19: -68.3848419, 48.4514236, -68.5800323, 48.5331650, -116.9179993, 117.0314560
20: -60.8479576, 52.7312737, -61.0485764, 52.8692665, -113.7172241, 113.7798309
21: -84.5057297, 57.6774330, -84.7912292, 57.8504105, -142.3561401, 142.4686584
22: -85.2066956, 58.0677567, -85.4865265, 58.2718010, -143.4785004, 143.5542908
23: -69.9317780, 57.3499985, -70.1140442, 57.4580956, -127.3898621, 127.4640427
24: -78.8860855, 52.8611717, -79.1644745, 52.9946289, -131.8807068, 132.0256500
25: -72.6233521, 62.0051537, -72.7966003, 62.1567383, -134.7800903, 134.8017578
26: -98.2069397, 87.1182098, -98.5211105, 87.4056549, -185.6125946, 185.6393127
27: -85.2542801, 58.8834457, -85.6042938, 59.0702019, -144.3244781, 144.4877319
28: -69.6882782, 63.5879440, -69.8646393, 63.7158699, -133.4041443, 133.4525757
29: -88.2226257, 50.9105492, -88.4381790, 51.0930901, -139.3157196, 139.3487091
30: -86.8901520, 65.1539383, -87.0815201, 65.3586578, -152.2488098, 152.2354431
31: -84.8343201, 55.4123650, -85.1084595, 55.4954109, -140.3297272, 140.5208282
32: -75.5650330, 61.7727699, -75.7625885, 62.0012894, -137.5663147, 137.5353546
33: -107.5151291, 82.1626663, -107.8745651, 82.4537811, -189.9689026, 190.0372314
34: -87.4266281, 65.5697861, -87.6693954, 65.8384247, -153.2650452, 153.2391815
35: -83.5157547, 68.3576660, -83.7963791, 68.6121063, -152.1278381, 152.1540527
36: -82.6186447, 73.5060120, -82.8268280, 73.6401825, -156.2588196, 156.3328247
37: -123.0927048, 71.0162354, -123.4338684, 71.1833344, -194.2760315, 194.4500885
38: -100.9975967, 92.9542847, -101.2620773, 93.1344528, -194.1320496, 194.2163391
39: -115.1795654, 83.3827515, -115.4618607, 83.5188141, -198.6983795, 198.8446045
40: -98.3746643, 60.0926590, -98.7000809, 60.2917252, -158.6663666, 158.7927399
41: -78.5238266, 62.8973999, -78.7636948, 63.0321808, -141.5559998, 141.6611023
42: -63.3596535, 58.6848717, -63.5771446, 58.9491692, -122.3088074, 122.2620163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=448, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=635, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.7067989, upper bound: 112.7177280
time: 163.15 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.7067989, upper bound: 112.7185355
time: 82.27 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -106.1341934, 77.0433197, -106.1548843, 77.0507812, -183.1849670, 183.1982117
1: -62.9125290, 57.7340736, -62.9262085, 57.7415848, -120.6541138, 120.6602783
2: -58.6830139, 58.2630463, -58.6994400, 58.2702408, -116.9532471, 116.9624863
3: -68.0199890, 66.9307556, -68.0401840, 66.9403763, -134.9603577, 134.9709320
4: -73.5132523, 69.3889694, -73.5379944, 69.3985825, -142.9118347, 142.9269562
5: -70.3380203, 71.8568115, -70.3595276, 71.8663330, -142.2043457, 142.2163391
6: -81.5817413, 61.4390564, -81.5971909, 61.4657860, -143.0475311, 143.0362549
7: -77.9623184, 65.0399628, -77.9824371, 65.0511780, -143.0134888, 143.0223999
8: -81.6675262, 78.8113861, -81.6882324, 78.8199463, -160.4874725, 160.4996033
9: -71.8656158, 68.6264420, -71.8765488, 68.6490173, -140.5146179, 140.5029907
10: -98.1372833, 86.8112640, -98.1530457, 86.8526764, -184.9899292, 184.9643097
11: -92.9973526, 65.7273712, -93.0088806, 65.7568207, -158.7541809, 158.7362518
12: -87.0223999, 89.9753723, -87.0343323, 90.0132294, -177.0356293, 177.0097046
13: -90.5028687, 98.5130844, -90.5174255, 98.5364304, -189.0393066, 189.0305176
14: -133.8108521, 76.4723053, -133.8308563, 76.5036087, -210.3144379, 210.3031616
15: -85.2975311, 64.2654114, -85.3244095, 64.2747879, -149.5723267, 149.5898132
16: -101.7602463, 69.2506409, -101.7769470, 69.2779999, -171.0382385, 171.0275879
17: -136.1045074, 82.0433655, -136.1198425, 82.0640030, -218.1684875, 218.1632080
18: -85.8267746, 69.0953674, -85.8512650, 69.1128082, -154.9395752, 154.9466248
19: -68.6331329, 48.5608444, -68.6562653, 48.5697403, -117.2028656, 117.2171097
20: -61.1089859, 52.9547348, -61.1167030, 52.9678955, -114.0768738, 114.0714417
21: -84.8646088, 57.9534798, -84.8766251, 57.9678040, -142.8324127, 142.8300781
22: -85.6066132, 58.3436966, -85.6397552, 58.3579597, -143.9645691, 143.9834595
23: -70.1796722, 57.4995766, -70.1915894, 57.5139618, -127.6936340, 127.6911621
24: -79.3418427, 53.0184937, -79.3620071, 53.0258827, -132.3677216, 132.3804932
25: -72.8784790, 62.2197151, -72.8944855, 62.2333145, -135.1117859, 135.1141968
26: -98.6059189, 87.5881500, -98.6203003, 87.6083450, -186.2142334, 186.2084503
27: -85.8276062, 59.0948372, -85.8516464, 59.1053925, -144.9329987, 144.9464722
28: -69.9539413, 63.7523041, -69.9653625, 63.7628822, -133.7168274, 133.7176666
29: -88.5135345, 51.1854858, -88.5352859, 51.2017708, -139.7152710, 139.7207642
30: -87.1540222, 65.4814453, -87.1652298, 65.5019073, -152.6559296, 152.6466675
31: -85.2207870, 55.5220871, -85.2425232, 55.5306664, -140.7514343, 140.7646179
32: -75.8331146, 62.1546898, -75.8450851, 62.1739235, -138.0070343, 137.9997711
33: -108.1132355, 82.5019226, -108.1417618, 82.5120392, -190.6252747, 190.6436768
34: -87.8157349, 65.8787384, -87.8366394, 65.8918991, -153.7076263, 153.7153778
35: -83.9834976, 68.6544952, -84.0078888, 68.6628647, -152.6463623, 152.6623688
36: -82.9299316, 73.6796112, -82.9497986, 73.6886520, -156.6185913, 156.6294098
37: -123.6186905, 71.2336426, -123.6477203, 71.2431335, -194.8618164, 194.8813629
38: -101.3820648, 93.1789780, -101.4057236, 93.1960754, -194.5781403, 194.5846863
39: -115.5723572, 83.5531464, -115.6087418, 83.5611115, -199.1334686, 199.1618652
40: -98.8797226, 60.3155746, -98.8976135, 60.3231430, -159.2028656, 159.2131958
41: -78.8964462, 63.0787048, -78.9140472, 63.0916748, -141.9881287, 141.9927521
42: -63.6430740, 59.0981865, -63.6539116, 59.1306686, -122.7737427, 122.7520981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=448, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=636, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

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
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.7067989, upper bound: 112.8831563
time: 92.94 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.7067989, upper bound: 112.8837383
time: 92.80 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 188.08 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 188.08
Output dim: 12, lower bound: -112.7067989, upper bound: 112.7177280
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 188.08
Output dim: 12, lower bound: -112.7067989, upper bound: 112.7185355
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 188.08
Output dim: 12, lower bound: -112.7067989, upper bound: 112.8831563
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 188.08
Output dim: 12, lower bound: -112.7067989, upper bound: 112.8837383

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -105.5717392, 76.7721558, -105.5753326, 76.6267548, -182.1984863, 182.3474884
1: -62.5710564, 57.5646896, -62.5591125, 57.4570961, -120.0281525, 120.1238022
2: -58.0898743, 58.0607109, -58.0754547, 57.9243965, -116.0142670, 116.1361694
3: -67.4444122, 66.6509094, -67.4090652, 66.5081329, -133.9525452, 134.0599670
4: -72.7877045, 69.1254120, -72.7917175, 68.9890137, -141.7767181, 141.9171295
5: -69.7516022, 71.5492096, -69.7114182, 71.3866272, -141.1382294, 141.2606201
6: -81.2086105, 61.0749588, -81.2415009, 61.1278610, -142.3364410, 142.3164673
7: -77.4350204, 64.8258057, -77.4176941, 64.6912537, -142.1262665, 142.2434998
8: -81.0865936, 78.5445175, -81.0988846, 78.4035950, -159.4901733, 159.6434021
9: -71.5539017, 67.9641342, -71.5294037, 68.0220871, -139.5759888, 139.4935303
10: -97.5054245, 85.5559235, -97.4505539, 85.6491241, -183.1545410, 183.0064697
11: -92.4736786, 64.8945847, -92.4735184, 64.9584808, -157.4321594, 157.3681030
12: -86.4363861, 88.7075195, -86.3351288, 88.7738953, -175.2102814, 175.0426483
13: -90.1694489, 97.8468857, -90.2822266, 97.9746780, -188.1441345, 188.1291046
14: -133.0744629, 75.3926849, -133.1657410, 75.5495377, -208.6239929, 208.5584259
15: -84.5699921, 63.8889885, -84.6259460, 63.9117126, -148.4816895, 148.5149231
16: -101.2934875, 68.5313492, -101.2803574, 68.6108627, -169.9043579, 169.8117065
17: -135.5215149, 81.1438446, -135.5881042, 81.2670441, -216.7885590, 216.7319183
18: -85.2406387, 68.7923584, -85.2896271, 68.8087311, -154.0493774, 154.0819855
19: -68.3281250, 48.3792686, -68.2881012, 48.3628540, -116.6909637, 116.6673737
20: -60.7935410, 52.6370468, -60.7799416, 52.6540565, -113.4476013, 113.4169922
21: -84.4480515, 57.5325584, -84.4048386, 57.5216560, -141.9697113, 141.9373932
22: -85.1428528, 57.9566803, -85.2487106, 57.9693832, -143.1122284, 143.2053833
23: -69.8818207, 57.2734604, -69.8720016, 57.2668114, -127.1486359, 127.1454620
24: -78.7975845, 52.8335419, -78.9156342, 52.9068947, -131.7044678, 131.7491760
25: -72.5765152, 61.9089699, -72.6121979, 61.9112968, -134.4878082, 134.5211639
26: -98.1394958, 86.8525543, -98.0168457, 86.8055801, -184.9450684, 184.8694000
27: -85.1277390, 58.8497162, -85.2633362, 58.9354286, -144.0631714, 144.1130524
28: -69.6394577, 63.5394402, -69.6610947, 63.5766563, -133.2161102, 133.2005310
29: -88.1727066, 50.7698059, -88.2021790, 50.7668648, -138.9395752, 138.9719849
30: -86.8422089, 65.0040436, -86.7973099, 65.0138245, -151.8560333, 151.8013306
31: -84.7507095, 55.3564224, -84.7657852, 55.3483124, -140.0990143, 140.1221924
32: -75.5107422, 61.6429939, -75.4978943, 61.7024422, -137.2131805, 137.1408691
33: -107.3432312, 82.1067505, -107.4638138, 82.1517792, -189.4950104, 189.5705566
34: -87.3133316, 65.5185089, -87.3757858, 65.5873718, -152.9006958, 152.8942871
35: -83.3740082, 68.3142090, -83.4677582, 68.3614044, -151.7354126, 151.7819672
36: -82.5444641, 73.4611053, -82.6127930, 73.4815979, -156.0260620, 156.0738983
37: -123.0033417, 70.9350433, -123.1328430, 70.9549942, -193.9583435, 194.0678711
38: -100.8701096, 92.9112854, -100.9105835, 92.8790283, -193.7491302, 193.8218689
39: -115.0855026, 83.3392181, -115.1542816, 83.3311462, -198.4166107, 198.4934845
40: -98.2816467, 60.0646057, -98.3989334, 60.1298332, -158.4114685, 158.4635315
41: -78.4474564, 62.8394165, -78.5327911, 62.8563194, -141.3037720, 141.3721924
42: -63.3073959, 58.5195847, -63.2947540, 58.5473289, -121.8547211, 121.8143311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=635, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6952289, upper bound: 112.5485115
time: 79.18 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6962663, upper bound: 112.7082848
time: 84.23 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -105.7301331, 76.8066559, -105.9718018, 76.9771957, -182.7073364, 182.7784576
1: -62.6778069, 57.5874443, -62.8214836, 57.6958427, -120.3736420, 120.4089279
2: -58.2770538, 58.0837784, -58.4973221, 58.2245941, -116.5016327, 116.5811005
3: -67.6413269, 66.6880951, -67.8513718, 66.8646317, -134.5059509, 134.5394592
4: -72.9997711, 69.1562805, -73.2728119, 69.3302612, -142.3300171, 142.4290771
5: -69.9541626, 71.5863266, -70.1744537, 71.7897263, -141.7438812, 141.7607727
6: -81.2713165, 61.1338158, -81.4806366, 61.3111343, -142.5824280, 142.6144562
7: -77.5948639, 64.8535614, -77.8296356, 64.9876099, -142.5824585, 142.6831970
8: -81.2527924, 78.5788574, -81.4888916, 78.7453308, -159.9981232, 160.0677490
9: -71.6022949, 68.1188660, -71.8059921, 68.3950806, -139.9973755, 139.9248657
10: -97.5836639, 85.8783798, -98.0346451, 86.3659973, -183.9496613, 183.9130249
11: -92.5247345, 65.1141205, -92.9125519, 65.4391937, -157.9639282, 158.0266724
12: -86.4765701, 89.0569839, -86.9515381, 89.5418549, -176.0184174, 176.0085144
13: -90.2170486, 97.9420624, -90.4419327, 98.2758102, -188.4928589, 188.3839722
14: -133.1623535, 75.6282196, -133.6972198, 76.0701828, -209.2325287, 209.3254242
15: -84.7491531, 63.9410515, -85.0579453, 64.1841125, -148.9332581, 148.9989929
16: -101.3691330, 68.6917038, -101.6508942, 69.0000992, -170.3692169, 170.3425903
17: -135.5739746, 81.3304443, -136.0350952, 81.7195587, -217.2935181, 217.3655396
18: -85.3369598, 68.8691101, -85.6496429, 69.0194092, -154.3563538, 154.5187531
19: -68.3788300, 48.4448318, -68.5681610, 48.5212898, -116.9001160, 117.0129929
20: -60.8419724, 52.7221184, -61.0373383, 52.8513145, -113.6932831, 113.7594452
21: -84.4985809, 57.6651726, -84.7780151, 57.8261642, -142.3247375, 142.4431915
22: -85.1986771, 58.0515900, -85.4714203, 58.2402496, -143.4389343, 143.5230103
23: -69.9260864, 57.3380356, -70.1032410, 57.4384041, -127.3644714, 127.4412766
24: -78.8746643, 52.8569641, -79.1419678, 52.9872894, -131.8619537, 131.9989166
25: -72.6171341, 61.9962044, -72.7846603, 62.1395035, -134.7566376, 134.7808685
26: -98.1987610, 87.0939789, -98.5058594, 87.3661194, -185.5648499, 185.5998383
27: -85.2415695, 58.8782349, -85.5791702, 59.0605125, -144.3020782, 144.4573975
28: -69.6832428, 63.5789337, -69.8548508, 63.6986198, -133.3818665, 133.4337769
29: -88.2147217, 50.8950119, -88.4234848, 51.0652542, -139.2799683, 139.3184967
30: -86.8834915, 65.1404419, -87.0688248, 65.3325958, -152.2160645, 152.2092590
31: -84.8257828, 55.4056129, -85.0917587, 55.4826851, -140.3084717, 140.4973602
32: -75.5580597, 61.7631721, -75.7495804, 61.9824104, -137.5404663, 137.5127563
33: -107.4988251, 82.1565552, -107.8430405, 82.4420624, -189.9408875, 189.9996033
34: -87.4156952, 65.5634460, -87.6482925, 65.8262024, -153.2418823, 153.2117310
35: -83.5015335, 68.3534698, -83.7687225, 68.6037292, -152.1052551, 152.1221771
36: -82.6071701, 73.5007706, -82.8048019, 73.6306763, -156.2378235, 156.3055725
37: -123.0803528, 71.0047989, -123.4107819, 71.1605377, -194.2408600, 194.4155884
38: -100.9836578, 92.9490204, -101.2353058, 93.1239548, -194.1076050, 194.1843109
39: -115.1680374, 83.3779144, -115.4397812, 83.5094452, -198.6774597, 198.8176880
40: -98.3635483, 60.0880356, -98.6791229, 60.2828026, -158.6463470, 158.7671509
41: -78.5152664, 62.8851585, -78.7476501, 63.0074158, -141.5226746, 141.6328125
42: -63.3530235, 58.6644249, -63.5651779, 58.9140778, -122.2671051, 122.2295990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=635, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1315

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6952289, upper bound: 112.5490887
time: 89.55 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6962663, upper bound: 112.7089607
time: 105.01 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -105.9576035, 77.0044098, -105.7232513, 76.6930084, -182.6506042, 182.7276459
1: -62.7939873, 57.7079773, -62.6404114, 57.4975586, -120.2915497, 120.3483810
2: -58.4772987, 58.2366867, -58.2450676, 57.9639359, -116.4412384, 116.4817352
3: -67.8016357, 66.8875885, -67.5602798, 66.5732117, -134.3748474, 134.4478607
4: -73.2820740, 69.3523254, -73.0196457, 69.0476379, -142.3297119, 142.3719788
5: -70.1153412, 71.8145294, -69.8598175, 71.4535522, -141.5688934, 141.6743469
6: -81.5092545, 61.3586273, -81.3404846, 61.2417450, -142.7509766, 142.6990967
7: -77.7828903, 65.0073471, -77.5359421, 64.7473831, -142.5302734, 142.5432892
8: -81.4856949, 78.7720108, -81.2672729, 78.4691620, -159.9548645, 160.0392761
9: -71.8108063, 68.4560852, -71.5876617, 68.2454376, -140.0562439, 140.0437469
10: -98.0499878, 86.4589005, -97.5513763, 86.0751266, -184.1251221, 184.0102844
11: -92.9389801, 65.4866104, -92.5562820, 65.2350769, -158.1740570, 158.0428925
12: -86.9759521, 89.5958939, -86.4065857, 89.1856537, -176.1616058, 176.0024719
13: -90.4458618, 98.4007263, -90.3411789, 98.2040558, -188.6499176, 188.7419128
14: -133.7110596, 76.2144318, -133.2778931, 75.9425659, -209.6536255, 209.4923248
15: -85.0981445, 64.2061920, -84.8545380, 63.9882545, -149.0863953, 149.0607300
16: -101.6739578, 69.0721359, -101.3861923, 68.8535767, -170.5275269, 170.4583130
17: -136.0441284, 81.8364410, -135.6590881, 81.5800552, -217.6241760, 217.4954987
18: -85.7157059, 69.0060577, -85.4642334, 68.8785324, -154.5942383, 154.4702911
19: -68.5756989, 48.4889755, -68.3680038, 48.3992386, -116.9749298, 116.8569794
20: -61.0546303, 52.8601837, -60.8485489, 52.7522736, -113.8069000, 113.7087250
21: -84.8071899, 57.8087425, -84.4905014, 57.6388702, -142.4460449, 142.2992401
22: -85.5414505, 58.2321815, -85.4014511, 58.0548782, -143.5963287, 143.6336365
23: -70.1290588, 57.4192123, -69.9499130, 57.3195381, -127.4485703, 127.3691254
24: -79.2515717, 52.9902878, -79.1120453, 52.9374733, -132.1890411, 132.1023254
25: -72.8308258, 62.1230621, -72.7121429, 61.9872894, -134.8181152, 134.8352051
26: -98.5377884, 87.3138733, -98.1166000, 86.9996033, -185.5373840, 185.4304810
27: -85.7002411, 59.0609207, -85.5101776, 58.9703217, -144.6705627, 144.5711060
28: -69.9042130, 63.7032890, -69.7628250, 63.6229858, -133.5271912, 133.4661102
29: -88.4624557, 51.0440559, -88.3028870, 50.8743858, -139.3368225, 139.3469391
30: -87.1052628, 65.3309479, -86.8820877, 65.1564484, -152.2617035, 152.2130432
31: -85.1356201, 55.4657974, -84.8995590, 55.3831329, -140.5187531, 140.3653564
32: -75.7785339, 62.0236855, -75.5805664, 61.8751640, -137.6537018, 137.6042480
33: -107.9403534, 82.4461365, -107.7301865, 82.2103348, -190.1506805, 190.1763153
34: -87.7016373, 65.8275757, -87.5421677, 65.6414871, -153.3431244, 153.3697357
35: -83.8408966, 68.6115646, -83.6785660, 68.4130554, -152.2539520, 152.2901306
36: -82.8547974, 73.6351318, -82.7348328, 73.5291824, -156.3839569, 156.3699646
37: -123.5275421, 71.1534271, -123.3456955, 71.0140305, -194.5415649, 194.4990997
38: -101.2539597, 93.1352234, -101.0533752, 92.9421997, -194.1961670, 194.1885986
39: -115.4773712, 83.5093842, -115.3012695, 83.3735199, -198.8508911, 198.8106537
40: -98.7854767, 60.2873039, -98.5953903, 60.1615448, -158.9470215, 158.8826904
41: -78.8191986, 63.0216980, -78.6841660, 62.9161949, -141.7353821, 141.7058716
42: -63.5906525, 58.9229584, -63.3732071, 58.7223015, -122.3129578, 122.2961578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=636, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6952289, upper bound: 112.7183551
time: 93.25 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6962663, upper bound: 112.8724286
time: 79.71 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -106.1173401, 77.0394745, -106.1227417, 77.0436249, -183.1609650, 183.1622162
1: -62.9004478, 57.7314110, -62.9030151, 57.7367287, -120.6371765, 120.6344299
2: -58.6697121, 58.2599220, -58.6729164, 58.2644157, -116.9341278, 116.9328232
3: -68.0040894, 66.9254608, -68.0083618, 66.9304047, -134.9344940, 134.9338074
4: -73.4941940, 69.3840179, -73.5015411, 69.3894882, -142.8836823, 142.8855591
5: -70.3228455, 71.8519440, -70.3310471, 71.8569641, -142.1798096, 142.1829834
6: -81.5729065, 61.4193039, -81.5806732, 61.4254036, -142.9983063, 142.9999695
7: -77.9446411, 65.0362167, -77.9520416, 65.0441132, -142.9887543, 142.9882507
8: -81.6524887, 78.8067627, -81.6596222, 78.8112030, -160.4636841, 160.4663849
9: -71.8594971, 68.6109467, -71.8650818, 68.6192322, -140.4787292, 140.4760284
10: -98.1280823, 86.7802124, -98.1356277, 86.7927856, -184.9208679, 184.9158325
11: -92.9900360, 65.7068939, -92.9950104, 65.7176819, -158.7077026, 158.7019043
12: -87.0167542, 89.9475327, -87.0238419, 89.9563141, -176.9730225, 176.9713593
13: -90.4948349, 98.4978180, -90.5028305, 98.5066299, -189.0014648, 189.0006409
14: -133.7997742, 76.4572601, -133.8099670, 76.4678879, -210.2676697, 210.2672119
15: -85.2790070, 64.2579422, -85.2897797, 64.2607880, -149.5397949, 149.5477295
16: -101.7502136, 69.2334747, -101.7576904, 69.2456894, -170.9959106, 170.9911499
17: -136.0976562, 82.0291290, -136.1072388, 82.0365448, -218.1342010, 218.1363678
18: -85.8136063, 69.0837021, -85.8258667, 69.0910950, -154.9046936, 154.9095764
19: -68.6271820, 48.5548058, -68.6449432, 48.5585251, -117.1857071, 117.1997528
20: -61.1033630, 52.9456673, -61.1061363, 52.9504623, -114.0538254, 114.0518036
21: -84.8577957, 57.9410744, -84.8638916, 57.9440384, -142.8018188, 142.8049622
22: -85.5992050, 58.3275070, -85.6257477, 58.3265343, -143.9257355, 143.9532471
23: -70.1742249, 57.4902306, -70.1812286, 57.4962540, -127.6704788, 127.6714630
24: -79.3315887, 53.0147057, -79.3406906, 53.0190926, -132.3506622, 132.3553925
25: -72.8726425, 62.2114105, -72.8834610, 62.2179832, -135.0906219, 135.0948792
26: -98.5984497, 87.5690308, -98.6065216, 87.5715637, -186.1700134, 186.1755371
27: -85.8149796, 59.0897980, -85.8270569, 59.0961494, -144.9111328, 144.9168549
28: -69.9490356, 63.7434425, -69.9559784, 63.7463684, -133.6954041, 133.6994171
29: -88.5063095, 51.1699638, -88.5217209, 51.1745796, -139.6808777, 139.6916809
30: -87.1476593, 65.4682312, -87.1531982, 65.4770050, -152.6246643, 152.6214294
31: -85.2122879, 55.5155449, -85.2263031, 55.5187798, -140.7310638, 140.7418518
32: -75.8264771, 62.1478271, -75.8325653, 62.1589699, -137.9854431, 137.9803925
33: -108.0973129, 82.4959564, -108.1110992, 82.5008163, -190.5981293, 190.6070557
34: -87.8048325, 65.8725891, -87.8156662, 65.8806076, -153.6854401, 153.6882629
35: -83.9694901, 68.6502533, -83.9808731, 68.6546783, -152.6241608, 152.6311340
36: -82.9188156, 73.6743164, -82.9285583, 73.6785583, -156.5973816, 156.6028595
37: -123.6073151, 71.2219772, -123.6263123, 71.2209625, -194.8282623, 194.8482971
38: -101.3681412, 93.1737518, -101.3798752, 93.1860809, -194.5542297, 194.5536194
39: -115.5614395, 83.5482330, -115.5880814, 83.5518570, -199.1132965, 199.1362915
40: -98.8692551, 60.3110886, -98.8774796, 60.3146667, -159.1839294, 159.1885681
41: -78.8881302, 63.0659523, -78.8984528, 63.0675583, -141.9556732, 141.9644012
42: -63.6368370, 59.0789642, -63.6422729, 59.0969620, -122.7337799, 122.7212372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=636, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6952289, upper bound: 112.7192504
time: 89.88 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6962663, upper bound: 112.8730508
time: 97.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 189.33 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 189.33
Output dim: 12, lower bound: -112.6952289, upper bound: 112.5485115
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 189.33
Output dim: 12, lower bound: -112.6962663, upper bound: 112.7082848
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 189.33
Output dim: 12, lower bound: -112.6952289, upper bound: 112.5490887
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 189.33
Output dim: 12, lower bound: -112.6962663, upper bound: 112.7089607
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 189.33
Output dim: 12, lower bound: -112.6952289, upper bound: 112.7183551
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 189.33
Output dim: 12, lower bound: -112.6962663, upper bound: 112.8724286
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 189.33
Output dim: 12, lower bound: -112.6952289, upper bound: 112.7192504
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 189.33
Output dim: 12, lower bound: -112.6962663, upper bound: 112.8730508

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -105.1850815, 76.5760956, -105.4074097, 76.5728531, -181.7579346, 181.9834900
1: -62.3366699, 57.4470711, -62.4586258, 57.4206543, -119.7573166, 119.9057007
2: -57.6790428, 57.8915787, -57.8798256, 57.8899498, -115.5689926, 115.7714081
3: -67.0345459, 66.4387054, -67.2151718, 66.4502640, -133.4848022, 133.6538696
4: -72.3089447, 68.9411774, -72.5626831, 68.9405518, -141.2494965, 141.5038605
5: -69.3358612, 71.2791748, -69.5110168, 71.3259201, -140.6617737, 140.7901917
6: -80.9824829, 60.7493172, -81.1637573, 60.9798737, -141.9623413, 141.9130707
7: -77.0544739, 64.6712952, -77.2528534, 64.6432266, -141.6976929, 141.9241333
8: -80.6781158, 78.3561096, -80.9082489, 78.3496246, -159.0277252, 159.2643585
9: -71.4012527, 67.5262985, -71.4697571, 67.8271866, -139.2284241, 138.9960480
10: -97.2520599, 84.8396301, -97.3620453, 85.3174820, -182.5695038, 182.2016754
11: -92.1559219, 64.3747635, -92.3943100, 64.7101212, -156.8660278, 156.7690735
12: -85.9843750, 87.7688293, -86.2707367, 88.3208923, -174.3052368, 174.0395660
13: -89.9258881, 97.2999649, -90.2188416, 97.7312317, -187.6571198, 187.5187988
14: -132.5762634, 74.6760559, -133.0452576, 75.2013702, -207.7776031, 207.7212830
15: -84.0729523, 63.6374931, -84.3942719, 63.8447990, -147.9177551, 148.0317688
16: -100.9300995, 68.0909271, -101.1473694, 68.4141083, -169.3441772, 169.2382812
17: -135.0661011, 80.4497604, -135.5063171, 80.9355011, -216.0016022, 215.9560852
18: -84.8231659, 68.5582275, -85.1162109, 68.7121277, -153.5352936, 153.6744385
19: -68.0957718, 48.2413635, -68.2061310, 48.3026505, -116.3984070, 116.4474792
20: -60.5850410, 52.3992004, -60.7146072, 52.5461235, -113.1311493, 113.1138077
21: -84.1797562, 57.2418518, -84.3270874, 57.3901711, -141.5699310, 141.5689392
22: -84.8475494, 57.6537018, -85.1311874, 57.8475304, -142.6950836, 142.7848816
23: -69.6426086, 57.1320724, -69.7865601, 57.2091904, -126.8517990, 126.9186325
24: -78.3903275, 52.7143517, -78.7359238, 52.8750572, -131.2653809, 131.4502563
25: -72.3159180, 61.6872864, -72.5044708, 61.8193817, -134.1352997, 134.1917572
26: -97.7973862, 86.3269958, -97.9239426, 86.5597534, -184.3571472, 184.2509308
27: -84.6580429, 58.7274132, -85.0551147, 58.9021225, -143.5601501, 143.7825317
28: -69.4183807, 63.4304848, -69.5796356, 63.5307312, -132.9490967, 133.0101166
29: -87.9426270, 50.4327087, -88.1192703, 50.6119194, -138.5545349, 138.5519714
30: -86.6049271, 64.7204285, -86.7113876, 64.8852997, -151.4902191, 151.4318237
31: -84.3749771, 55.2429924, -84.6190186, 55.3039207, -139.6788940, 139.8620148
32: -75.2833405, 61.2152100, -75.4308929, 61.4987144, -136.7820435, 136.6460876
33: -106.7934494, 81.9077759, -107.2238693, 82.0890274, -188.8824768, 189.1316528
34: -87.0119553, 65.3343964, -87.2480545, 65.5351257, -152.5470581, 152.5824280
35: -83.0099030, 68.1633911, -83.3113251, 68.3119659, -151.3218689, 151.4747162
36: -82.3436584, 73.2259369, -82.5486298, 73.3785095, -155.7221680, 155.7745667
37: -122.5967331, 70.7331238, -122.9751587, 70.8711624, -193.4678955, 193.7082825
38: -100.5337906, 92.6016693, -100.7996445, 92.7449493, -193.2787476, 193.4013062
39: -114.7304916, 83.1305389, -115.0249176, 83.2521591, -197.9826508, 198.1554565
40: -97.8619232, 59.9240913, -98.2257843, 60.0943985, -157.9563293, 158.1498718
41: -78.1831894, 62.6286469, -78.4280319, 62.7677040, -140.9508972, 141.0566711
42: -63.1052704, 58.1040077, -63.2346344, 58.3532753, -121.4585419, 121.3386383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=446, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=634, inp2_unstable=636, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

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
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5710057, upper bound: 112.5422846
time: 74.08 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6863782, upper bound: 112.5422846
time: 79.99 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -105.5429459, 76.7633820, -105.5591049, 76.6222992, -182.1652527, 182.3224792
1: -62.5506554, 57.5554276, -62.5483322, 57.4523468, -120.0030060, 120.1037521
2: -58.0594101, 58.0537415, -58.0601807, 57.9208641, -115.9802704, 116.1139069
3: -67.4188538, 66.6404724, -67.3960114, 66.5028534, -133.9217072, 134.0364685
4: -72.7512665, 69.1144562, -72.7739258, 68.9833908, -141.7346497, 141.8883667
5: -69.7246475, 71.5393219, -69.6953964, 71.3816147, -141.1062469, 141.2347107
6: -81.1942215, 61.0453873, -81.2340698, 61.1136093, -142.3078156, 142.2794495
7: -77.4077072, 64.8159027, -77.4024200, 64.6861420, -142.0938416, 142.2183228
8: -81.0551376, 78.5343475, -81.0832520, 78.3983612, -159.4534912, 159.6175995
9: -71.5437927, 67.9325104, -71.5237503, 68.0067902, -139.5505829, 139.4562531
10: -97.4879456, 85.5064316, -97.4416122, 85.6251068, -183.1130371, 182.9480286
11: -92.4583054, 64.8555298, -92.4657364, 64.9388809, -157.3971863, 157.3212585
12: -86.4239883, 88.6453400, -86.3286667, 88.7431030, -175.1670837, 174.9739990
13: -90.1550903, 97.8102798, -90.2743607, 97.9567184, -188.1118011, 188.0846405
14: -133.0490112, 75.3489075, -133.1528015, 75.5282669, -208.5772705, 208.5017090
15: -84.5293427, 63.8766060, -84.6053391, 63.9055786, -148.4349060, 148.4819336
16: -101.2587051, 68.4947739, -101.2625961, 68.5929413, -169.8516388, 169.7573547
17: -135.5045471, 81.0992126, -135.5794373, 81.2449646, -216.7495117, 216.6786499
18: -85.2069244, 68.7681885, -85.2730331, 68.7962952, -154.0032043, 154.0411987
19: -68.2986603, 48.3684921, -68.2728577, 48.3573074, -116.6559677, 116.6413422
20: -60.7833214, 52.6183128, -60.7745934, 52.6446686, -113.4279861, 113.3928986
21: -84.4335556, 57.5113258, -84.3975906, 57.5108833, -141.9444427, 141.9089050
22: -85.1049423, 57.9303093, -85.2299652, 57.9556885, -143.0606384, 143.1602783
23: -69.8579025, 57.2571907, -69.8600311, 57.2585678, -127.1164627, 127.1172180
24: -78.7657700, 52.8273468, -78.9001923, 52.9028702, -131.6686249, 131.7275391
25: -72.5517578, 61.8952408, -72.5998383, 61.9038086, -134.4555664, 134.4950867
26: -98.1222382, 86.8164902, -98.0081177, 86.7862930, -184.9085388, 184.8246155
27: -85.0938873, 58.8399658, -85.2467575, 58.9302521, -144.0241394, 144.0867310
28: -69.6187897, 63.5261421, -69.6516800, 63.5699234, -133.1887207, 133.1778107
29: -88.1471252, 50.7532349, -88.1891098, 50.7577782, -138.9049072, 138.9423523
30: -86.8157806, 64.9739761, -86.7841339, 64.9976654, -151.8134460, 151.7580872
31: -84.7236023, 55.3440247, -84.7516785, 55.3416214, -140.0652161, 140.0957031
32: -75.4988251, 61.6100540, -75.4915161, 61.6863327, -137.1851501, 137.1015625
33: -107.3066406, 82.0955582, -107.4456863, 82.1449890, -189.4516144, 189.5412292
34: -87.2910385, 65.5065918, -87.3646545, 65.5802917, -152.8713226, 152.8712463
35: -83.3465958, 68.2978058, -83.4542007, 68.3526764, -151.6992798, 151.7520142
36: -82.5326462, 73.4449921, -82.6065674, 73.4730988, -156.0057373, 156.0515594
37: -122.9625778, 70.9152145, -123.1129913, 70.9446259, -193.9071960, 194.0281982
38: -100.8487854, 92.8912048, -100.8996506, 92.8704681, -193.7192535, 193.7908478
39: -115.0583420, 83.3235474, -115.1406937, 83.3222504, -198.3805847, 198.4642334
40: -98.2583237, 60.0546608, -98.3871002, 60.1241913, -158.3825073, 158.4417572
41: -78.4227829, 62.8214264, -78.5202789, 62.8474159, -141.2702026, 141.3417053
42: -63.2972488, 58.4806786, -63.2892838, 58.5278702, -121.8251114, 121.7699585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=446, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=635, inp2_unstable=636, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5717461, upper bound: 112.6989059
time: 97.16 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5717461, upper bound: 112.6989059
time: 121.40 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -105.3435516, 76.6107941, -105.8035812, 76.9234467, -182.2669983, 182.4143677
1: -62.4434738, 57.4697685, -62.7207947, 57.6596642, -120.1031342, 120.1905670
2: -57.8664207, 57.9149208, -58.3018684, 58.1903572, -116.0567780, 116.2167892
3: -67.2315216, 66.4759979, -67.6572571, 66.8073120, -134.0388336, 134.1332397
4: -72.5211639, 68.9721985, -73.0437775, 69.2820282, -141.8031921, 142.0159760
5: -69.5384216, 71.3166351, -69.9739075, 71.7295227, -141.2679291, 141.2905426
6: -81.0453186, 60.8059006, -81.4028931, 61.1611557, -142.2064667, 142.2088013
7: -77.2141571, 64.6990662, -77.6646118, 64.9397964, -142.1539612, 142.3636780
8: -80.8443604, 78.3906403, -81.2981415, 78.6916962, -159.5360260, 159.6887817
9: -71.4501495, 67.6809387, -71.7469406, 68.2001190, -139.6502686, 139.4278870
10: -97.3304977, 85.1623764, -97.9468002, 86.0343475, -183.3648376, 183.1091766
11: -92.2075195, 64.5945511, -92.8339767, 65.1908646, -157.3983765, 157.4285278
12: -86.0248260, 88.1183243, -86.8873901, 89.0886993, -175.1135254, 175.0057068
13: -89.9734497, 97.3942871, -90.3786163, 98.0321808, -188.0056305, 187.7729034
14: -132.6645966, 74.9113770, -133.5773468, 75.7218552, -208.3864441, 208.4887085
15: -84.2521210, 63.6897659, -84.8260956, 64.1176147, -148.3697357, 148.5158539
16: -101.0052032, 68.2508087, -101.5190125, 68.8025208, -169.8077087, 169.7697906
17: -135.1188354, 80.6360321, -135.9535828, 81.3876266, -216.5064697, 216.5896149
18: -84.9184113, 68.6349640, -85.4757233, 68.9225769, -153.8409882, 154.1106873
19: -68.1457977, 48.3071365, -68.4866486, 48.4610138, -116.6068115, 116.7937851
20: -60.6337967, 52.4843407, -60.9725342, 52.7432976, -113.3770905, 113.4568634
21: -84.2304916, 57.3746605, -84.7006454, 57.6946793, -141.9251556, 142.0753021
22: -84.9030914, 57.7479668, -85.3539734, 58.1181450, -143.0212402, 143.1019287
23: -69.6864166, 57.1967392, -70.0183411, 57.3805161, -127.0669327, 127.2150803
24: -78.4665527, 52.7378960, -78.9610825, 52.9555664, -131.4221191, 131.6989746
25: -72.3563538, 61.7743874, -72.6782379, 62.0474739, -134.4038239, 134.4526215
26: -97.8570404, 86.5685272, -98.4137115, 87.1197205, -184.9767609, 184.9822388
27: -84.7711487, 58.7560501, -85.3704834, 59.0272865, -143.7984314, 144.1265259
28: -69.4617310, 63.4700966, -69.7738800, 63.6527405, -133.1144714, 133.2439575
29: -87.9846115, 50.5576668, -88.3409882, 50.9105034, -138.8951111, 138.8986511
30: -86.6458893, 64.8564148, -86.9835358, 65.2040482, -151.8499451, 151.8399353
31: -84.4490814, 55.2922668, -84.9451294, 55.4382133, -139.8872986, 140.2373962
32: -75.3309479, 61.3353767, -75.6831894, 61.7786407, -137.1095886, 137.0185547
33: -106.9488525, 81.9579468, -107.6028671, 82.3796997, -189.3285522, 189.5608215
34: -87.1139832, 65.3795166, -87.5201492, 65.7744064, -152.8883972, 152.8996582
35: -83.1371689, 68.2025833, -83.6120911, 68.5546722, -151.6918335, 151.8146667
36: -82.4062347, 73.2653885, -82.7404404, 73.5281677, -155.9344025, 156.0058289
37: -122.6732330, 70.8029175, -123.2529068, 71.0760269, -193.7492371, 194.0558167
38: -100.6477203, 92.6390991, -101.1240158, 92.9920578, -193.6397400, 193.7630920
39: -114.8126068, 83.1691589, -115.3108673, 83.4307098, -198.2433167, 198.4800262
40: -97.9435120, 59.9475021, -98.5060959, 60.2474823, -158.1909943, 158.4535980
41: -78.2508316, 62.6744461, -78.6428070, 62.9185600, -141.1693878, 141.3172607
42: -63.1512260, 58.2482719, -63.5063858, 58.7185631, -121.8697891, 121.7546539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=446, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=634, inp2_unstable=636, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1672
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
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5710057, upper bound: 112.5429650
time: 84.25 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6863782, upper bound: 112.5429650
time: 90.52 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -105.7013016, 76.7978516, -105.9555054, 76.9726715, -182.6739807, 182.7533569
1: -62.6574020, 57.5780373, -62.8106728, 57.6910477, -120.3484344, 120.3887024
2: -58.2466278, 58.0767708, -58.4821396, 58.2209587, -116.4675598, 116.5588989
3: -67.6157837, 66.6775970, -67.8382416, 66.8592834, -134.4750519, 134.5158386
4: -72.9633179, 69.1452942, -73.2551270, 69.3246307, -142.2879486, 142.4004211
5: -69.9272156, 71.5764160, -70.1584473, 71.7846527, -141.7118530, 141.7348633
6: -81.2569275, 61.1044540, -81.4731293, 61.2968636, -142.5537872, 142.5775757
7: -77.5647278, 64.8436050, -77.8123703, 64.9823074, -142.5470276, 142.6559753
8: -81.2213516, 78.5686340, -81.4733276, 78.7400360, -159.9613647, 160.0419617
9: -71.5919952, 68.0872498, -71.8001175, 68.3799210, -139.9719086, 139.8873596
10: -97.5661240, 85.8288574, -98.0257721, 86.3420258, -183.9081421, 183.8546295
11: -92.5094223, 65.0749207, -92.9048080, 65.4194794, -157.9288940, 157.9797211
12: -86.4640884, 88.9948425, -86.9450684, 89.5111008, -175.9751892, 175.9398956
13: -90.2023697, 97.9056244, -90.4338074, 98.2577438, -188.4601135, 188.3394318
14: -133.1367950, 75.5843658, -133.6842194, 76.0488815, -209.1856689, 209.2685852
15: -84.7079468, 63.9287071, -85.0369263, 64.1779709, -148.8858948, 148.9656372
16: -101.3342056, 68.6548462, -101.6328964, 68.9820480, -170.3162384, 170.2877502
17: -135.5570221, 81.2857132, -136.0264587, 81.6972656, -217.2542572, 217.3121643
18: -85.3031006, 68.8448029, -85.6330795, 69.0067596, -154.3098450, 154.4778748
19: -68.3495026, 48.4340210, -68.5530243, 48.5157547, -116.8652420, 116.9870453
20: -60.8316917, 52.7033272, -61.0319595, 52.8418465, -113.6735306, 113.7352829
21: -84.4841080, 57.6438904, -84.7707367, 57.8153496, -142.2994537, 142.4146271
22: -85.1604614, 58.0251846, -85.4524078, 58.2264442, -143.3869019, 143.4775696
23: -69.9020691, 57.3216934, -70.0913315, 57.4301376, -127.3322067, 127.4130173
24: -78.8428268, 52.8506546, -79.1264496, 52.9830704, -131.8258972, 131.9771118
25: -72.5926971, 61.9823494, -72.7723160, 62.1318779, -134.7245483, 134.7546692
26: -98.1812897, 87.0580139, -98.4969177, 87.3468246, -185.5281067, 185.5549316
27: -85.2076263, 58.8684235, -85.5625381, 59.0553169, -144.2629395, 144.4309692
28: -69.6624603, 63.5656204, -69.8444519, 63.6918793, -133.3543396, 133.4100647
29: -88.1888885, 50.8758965, -88.4101639, 51.0545197, -139.2433929, 139.2860565
30: -86.8570175, 65.1102524, -87.0556488, 65.3164368, -152.1734467, 152.1658936
31: -84.7989349, 55.3930511, -85.0776367, 55.4758034, -140.2747345, 140.4706726
32: -75.5460968, 61.7302361, -75.7430878, 61.9663849, -137.5124817, 137.4733124
33: -107.4621506, 82.1451721, -107.8247528, 82.4350433, -189.8971863, 189.9699097
34: -87.3933792, 65.5513611, -87.6370163, 65.8189240, -153.2123108, 153.1883850
35: -83.4740143, 68.3369217, -83.7550278, 68.5950699, -152.0690918, 152.0919495
36: -82.5951385, 73.4846649, -82.7983246, 73.6222382, -156.2173767, 156.2829895
37: -123.0396347, 70.9848633, -123.3908157, 71.1501083, -194.1897430, 194.3756714
38: -100.9622726, 92.9287872, -101.2242813, 93.1136322, -194.0758972, 194.1530457
39: -115.1407166, 83.3621826, -115.4260406, 83.5005493, -198.6412354, 198.7882080
40: -98.3402710, 60.0780067, -98.6671448, 60.2771301, -158.6174011, 158.7451477
41: -78.4906464, 62.8673210, -78.7349930, 62.9985466, -141.4891968, 141.6023102
42: -63.3429031, 58.6256866, -63.5597687, 58.8945389, -122.2374420, 122.1854553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=446, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=635, inp2_unstable=636, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

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
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5717461, upper bound: 112.6997387
time: 115.36 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6869581, upper bound: 112.6997387
time: 99.36 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -105.5653839, 76.8086853, -105.5549088, 76.6390686, -182.2044525, 182.3635864
1: -62.5547981, 57.5906982, -62.5388222, 57.4611588, -120.0159607, 120.1295166
2: -58.0607643, 58.0668602, -58.0484734, 57.9296036, -115.9903717, 116.1153336
3: -67.3877258, 66.6755371, -67.3651352, 66.5156631, -133.9033813, 134.0406494
4: -72.7942429, 69.1669464, -72.7886047, 68.9992981, -141.7935181, 141.9555511
5: -69.6942596, 71.5450592, -69.6587677, 71.3930664, -141.0873260, 141.2038269
6: -81.2778854, 61.0163498, -81.2628326, 61.0890923, -142.3669739, 142.2791748
7: -77.3996353, 64.8533020, -77.3701172, 64.6994629, -142.0990906, 142.2234192
8: -81.0702744, 78.5829620, -81.0752563, 78.4153748, -159.4856567, 159.6582031
9: -71.6586380, 68.0125885, -71.5282822, 68.0494156, -139.7080383, 139.5408630
10: -97.7949600, 85.7325439, -97.4626846, 85.7414017, -183.5363159, 183.1952209
11: -92.6137619, 64.9592819, -92.4771347, 64.9853210, -157.5990753, 157.4364166
12: -86.5262756, 88.6494293, -86.3423157, 88.7310638, -175.2573395, 174.9917297
13: -90.2010498, 97.8511581, -90.2778168, 97.9599533, -188.1610107, 188.1289673
14: -133.2126465, 75.4964294, -133.1572723, 75.5940399, -208.8066711, 208.6537018
15: -84.5928421, 63.9469872, -84.6212769, 63.9215889, -148.5144196, 148.5682678
16: -101.3019562, 68.6238251, -101.2528763, 68.6549988, -169.9569550, 169.8766785
17: -135.5878448, 81.1388626, -135.5771332, 81.2478485, -216.8356934, 216.7159882
18: -85.2930984, 68.7693481, -85.2898483, 68.7814255, -154.0745239, 154.0592041
19: -68.3388824, 48.3488274, -68.2854691, 48.3386993, -116.6775742, 116.6342926
20: -60.8462944, 52.6197777, -60.7832451, 52.6440048, -113.4902954, 113.4030151
21: -84.5360565, 57.5160141, -84.4125824, 57.5070724, -142.0431213, 141.9285736
22: -85.2388535, 57.9245720, -85.2828369, 57.9324532, -143.1712952, 143.2074127
23: -69.8852997, 57.2747459, -69.8638916, 57.2616615, -127.1469574, 127.1386414
24: -78.8366776, 52.8716431, -78.9311066, 52.9056320, -131.7423096, 131.8027496
25: -72.5623169, 61.8987732, -72.6033020, 61.8951149, -134.4574127, 134.5020752
26: -98.1969833, 86.7875443, -98.0237732, 86.7544403, -184.9514160, 184.8113098
27: -85.2245560, 58.9349136, -85.3004913, 58.9368935, -144.1614380, 144.2354126
28: -69.6778717, 63.5922966, -69.6793137, 63.5769615, -133.2548218, 133.2716064
29: -88.2298050, 50.7045021, -88.2193756, 50.7191963, -138.9490051, 138.9238739
30: -86.8631058, 65.0378494, -86.7960587, 65.0261688, -151.8892822, 151.8339081
31: -84.7526093, 55.3508759, -84.7507401, 55.3387260, -140.0913391, 140.1016235
32: -75.5503998, 61.5888977, -75.5135803, 61.6704254, -137.2208252, 137.1024780
33: -107.3854752, 82.2471619, -107.4893646, 82.1476669, -189.5331421, 189.7365265
34: -87.3975296, 65.6403885, -87.4138565, 65.5893097, -152.9868469, 153.0542297
35: -83.4712982, 68.4578400, -83.5210724, 68.3632965, -151.8345947, 151.9789124
36: -82.6524963, 73.3968658, -82.6703796, 73.4256516, -156.0781555, 156.0672455
37: -123.1135406, 70.9496155, -123.1852264, 70.9297714, -194.0432892, 194.1348419
38: -100.9123459, 92.8224487, -100.9413376, 92.8067932, -193.7191467, 193.7637939
39: -115.1158371, 83.2976685, -115.1708069, 83.2944641, -198.4103088, 198.4684601
40: -98.3634796, 60.1451454, -98.4226685, 60.1261139, -158.4895782, 158.5677948
41: -78.5524673, 62.8066330, -78.5790939, 62.8268166, -141.3792877, 141.3857117
42: -63.3851433, 58.4944420, -63.3131104, 58.5243263, -121.9094696, 121.8075485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=446, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=635, inp2_unstable=636, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

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
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5710057, upper bound: 112.7120399
time: 81.83 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6863782, upper bound: 112.7120399
time: 95.92 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -105.9270630, 76.9957886, -105.7055893, 76.6885681, -182.6156311, 182.7013855
1: -62.7725143, 57.6986084, -62.6286354, 57.4927292, -120.2652435, 120.3272400
2: -58.4474792, 58.2299080, -58.2298317, 57.9604340, -116.4078979, 116.4597397
3: -67.7727737, 66.8772507, -67.5446014, 66.5680084, -134.3407898, 134.4218445
4: -73.2470551, 69.3416977, -73.0017548, 69.0420685, -142.2891083, 142.3434448
5: -70.0845642, 71.8047028, -69.8437805, 71.4485474, -141.5330811, 141.6484680
6: -81.4955139, 61.3315353, -81.3331223, 61.2279701, -142.7234802, 142.6646576
7: -77.7537766, 64.9974289, -77.5202637, 64.7422028, -142.4959717, 142.5177002
8: -81.4557190, 78.7618942, -81.2516937, 78.4639893, -159.9197083, 160.0135803
9: -71.8005981, 68.4255219, -71.5820312, 68.2302246, -140.0308228, 140.0075531
10: -98.0331573, 86.4107056, -97.5426178, 86.0510330, -184.0841827, 183.9533234
11: -92.9241562, 65.4481812, -92.5484848, 65.2152710, -158.1394348, 157.9966736
12: -86.9637146, 89.5336761, -86.4002075, 89.1546402, -176.1183472, 175.9338837
13: -90.4313354, 98.3648605, -90.3333435, 98.1862335, -188.6175690, 188.6982117
14: -133.6860657, 76.1708374, -133.2650452, 75.9212952, -209.6073608, 209.4358521
15: -85.0572968, 64.1943054, -84.8336334, 63.9822121, -149.0395050, 149.0279388
16: -101.6394348, 69.0362244, -101.3684158, 68.8355637, -170.4749451, 170.4046326
17: -136.0281219, 81.7918091, -135.6505432, 81.5578079, -217.5859070, 217.4423370
18: -85.6828003, 68.9820557, -85.4477005, 68.8661652, -154.5489655, 154.4297485
19: -68.5464706, 48.4784012, -68.3528824, 48.3937759, -116.9402313, 116.8312683
20: -61.0441628, 52.8418198, -60.8431778, 52.7429199, -113.7870789, 113.6849976
21: -84.7932281, 57.7876205, -84.4832306, 57.6281700, -142.4213867, 142.2708435
22: -85.5036011, 58.2060318, -85.3821640, 58.0413437, -143.5449524, 143.5881958
23: -70.1051407, 57.4034653, -69.9379120, 57.3113480, -127.4164734, 127.3413620
24: -79.2206039, 52.9842186, -79.0964203, 52.9334717, -132.1540833, 132.0806427
25: -72.8067551, 62.1090927, -72.6994781, 61.9798698, -134.7866211, 134.8085632
26: -98.5207977, 87.2741852, -98.1078644, 86.9782410, -185.4990082, 185.3820496
27: -85.6672516, 59.0517159, -85.4935837, 58.9652748, -144.6325073, 144.5453033
28: -69.8852463, 63.6906967, -69.7533112, 63.6163406, -133.5015869, 133.4440002
29: -88.4368134, 51.0244141, -88.2895050, 50.8641891, -139.3009949, 139.3139191
30: -87.0792465, 65.3027954, -86.8688278, 65.1402054, -152.2194519, 152.1716156
31: -85.1088638, 55.4532547, -84.8853683, 55.3765106, -140.4853821, 140.3386230
32: -75.7669983, 61.9914284, -75.5741882, 61.8588829, -137.6258850, 137.5656128
33: -107.9041290, 82.4349365, -107.7119904, 82.2036438, -190.1077576, 190.1469269
34: -87.6798248, 65.8159637, -87.5311508, 65.6345062, -153.3143005, 153.3471069
35: -83.8138428, 68.5956345, -83.6648865, 68.4042969, -152.2181396, 152.2605133
36: -82.8428040, 73.6194305, -82.7285919, 73.5208130, -156.3636169, 156.3480225
37: -123.4899826, 71.1338043, -123.3264542, 71.0036621, -194.4936371, 194.4602509
38: -101.2330246, 93.1153564, -101.0424347, 92.9337082, -194.1667175, 194.1577759
39: -115.4504089, 83.4936676, -115.2874069, 83.3647156, -198.8151093, 198.7810669
40: -98.7616119, 60.2777138, -98.5832672, 60.1559448, -158.9175568, 158.8609772
41: -78.7942810, 63.0045319, -78.6711731, 62.9073334, -141.7016144, 141.6756897
42: -63.5809059, 58.8904457, -63.3677521, 58.7057228, -122.2866287, 122.2581940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=446, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=636, inp2_unstable=636, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5717461, upper bound: 112.8617321
time: 76.53 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5717461, upper bound: 112.8617321
time: 113.20 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -105.7250137, 76.8439789, -105.9542007, 76.9899368, -182.7149353, 182.7981720
1: -62.6613541, 57.6139717, -62.8013039, 57.7004967, -120.3618469, 120.4152756
2: -58.2533455, 58.0902252, -58.4763107, 58.2302742, -116.4836197, 116.5665359
3: -67.5901871, 66.7136230, -67.8132782, 66.8732300, -134.4634094, 134.5269012
4: -73.0066605, 69.1988373, -73.2705078, 69.3413849, -142.3480377, 142.4693451
5: -69.9018784, 71.5827179, -70.1297836, 71.7969971, -141.6988678, 141.7124939
6: -81.3418427, 61.0762291, -81.5030212, 61.2727356, -142.6145630, 142.5792542
7: -77.5614777, 64.8821716, -77.7857208, 64.9965515, -142.5580292, 142.6678925
8: -81.2370453, 78.6178970, -81.4674835, 78.7577057, -159.9947510, 160.0853882
9: -71.7077637, 68.1675186, -71.8061676, 68.4231491, -140.1309204, 139.9736938
10: -97.8734131, 86.0542908, -98.0475464, 86.4590530, -184.3324585, 184.1018372
11: -92.6654510, 65.1798553, -92.9165726, 65.4678802, -158.1333160, 158.0964203
12: -86.5674133, 89.0010223, -86.9598160, 89.5016098, -176.0690308, 175.9608154
13: -90.2501373, 97.9486542, -90.4395218, 98.2622757, -188.5123901, 188.3881836
14: -133.3017883, 75.7390594, -133.6898499, 76.1191406, -209.4209290, 209.4289093
15: -84.7739258, 63.9989738, -85.0561523, 64.1945343, -148.9684448, 149.0551300
16: -101.3777466, 68.7849426, -101.6254883, 69.0461731, -170.4239197, 170.4104309
17: -135.6416931, 81.3312683, -136.0255737, 81.7037811, -217.3454590, 217.3568420
18: -85.3903122, 68.8468170, -85.6509705, 68.9935913, -154.3839111, 154.4977722
19: -68.3897095, 48.4147949, -68.5628052, 48.4979172, -116.8876266, 116.9776001
20: -60.8953400, 52.7053185, -61.0413551, 52.8420944, -113.7374268, 113.7466736
21: -84.5869370, 57.6485634, -84.7863235, 57.8121796, -142.3991089, 142.4348907
22: -85.2966003, 58.0193748, -85.5071869, 58.2040062, -143.5006104, 143.5265503
23: -69.9300461, 57.3459473, -70.0956497, 57.4379921, -127.3680420, 127.4415970
24: -78.9161530, 52.8961830, -79.1584625, 52.9873276, -131.9034576, 132.0546265
25: -72.6041260, 61.9869957, -72.7752914, 62.1255684, -134.7296906, 134.7622833
26: -98.2582626, 87.0425262, -98.5143661, 87.3261414, -185.5843811, 185.5568848
27: -85.3387070, 58.9639435, -85.6168365, 59.0628204, -144.4015198, 144.5807800
28: -69.7222900, 63.6325531, -69.8726196, 63.7003326, -133.4226074, 133.5051575
29: -88.2736206, 50.8302040, -88.4387131, 51.0191040, -139.2927094, 139.2689056
30: -86.9052048, 65.1750946, -87.0678024, 65.3466187, -152.2518005, 152.2428894
31: -84.8282013, 55.4006653, -85.0776367, 55.4741936, -140.3023834, 140.4782867
32: -75.5986328, 61.7132339, -75.7660828, 61.9541779, -137.5528107, 137.4793091
33: -107.5422287, 82.2973785, -107.8699646, 82.4385529, -189.9807739, 190.1673279
34: -87.5003433, 65.6857300, -87.6869812, 65.8286896, -153.3290253, 153.3727112
35: -83.5997086, 68.4964905, -83.8231506, 68.6055450, -152.2052612, 152.3196411
36: -82.7166061, 73.4362793, -82.8638992, 73.5757751, -156.2923737, 156.3001709
37: -123.1929550, 71.0177307, -123.4656982, 71.1361084, -194.3290710, 194.4834290
38: -101.0273056, 92.8604126, -101.2675018, 93.0540009, -194.0812988, 194.1279144
39: -115.1997757, 83.3363037, -115.4580917, 83.4730911, -198.6728516, 198.7943878
40: -98.4468384, 60.1688919, -98.7045059, 60.2792854, -158.7261200, 158.8733978
41: -78.6213379, 62.8507957, -78.7934799, 62.9779396, -141.5992737, 141.6442719
42: -63.4316788, 58.6501884, -63.5835152, 58.8983650, -122.3300323, 122.2337036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=446, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=635, inp2_unstable=636, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

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
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5710057, upper bound: 112.7128802
time: 87.79 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6863782, upper bound: 112.7128802
time: 91.25 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -106.0867310, 77.0307999, -106.1053467, 77.0390930, -183.1258240, 183.1361389
1: -62.8789825, 57.7218590, -62.8914223, 57.7318420, -120.6108246, 120.6132812
2: -58.6399765, 58.2531052, -58.6577873, 58.2608376, -116.9008102, 116.9108887
3: -67.9751968, 66.9150085, -67.9927216, 66.9250641, -134.9002686, 134.9077301
4: -73.4592438, 69.3734360, -73.4836731, 69.3839264, -142.8431396, 142.8571167
5: -70.2921753, 71.8420029, -70.3150787, 71.8518524, -142.1440277, 142.1570740
6: -81.5591736, 61.3920364, -81.5732574, 61.4114990, -142.9706726, 142.9652863
7: -77.9141235, 65.0261993, -77.9352417, 65.0388641, -142.9529877, 142.9614410
8: -81.6225281, 78.7966309, -81.6441116, 78.8060074, -160.4285278, 160.4407349
9: -71.8490906, 68.5803375, -71.8591766, 68.6040649, -140.4531555, 140.4395142
10: -98.1112595, 86.7319717, -98.1268311, 86.7686234, -184.8798676, 184.8587799
11: -92.9752274, 65.6683502, -92.9873352, 65.6976929, -158.6728973, 158.6556702
12: -87.0044403, 89.8853607, -87.0173645, 89.9253311, -176.9297791, 176.9027252
13: -90.4800873, 98.4620972, -90.4948120, 98.4888000, -188.9688873, 188.9569092
14: -133.7747040, 76.4136047, -133.7971039, 76.4464417, -210.2211304, 210.2107086
15: -85.2377625, 64.2460938, -85.2685013, 64.2547913, -149.4925537, 149.5145874
16: -101.7155609, 69.1972961, -101.7398911, 69.2274475, -170.9429779, 170.9371948
17: -136.0816650, 81.9844055, -136.0986633, 82.0142059, -218.0958557, 218.0830688
18: -85.7813339, 69.0594940, -85.8093033, 69.0784454, -154.8597717, 154.8687897
19: -68.5979156, 48.5441971, -68.6298141, 48.5530701, -117.1509857, 117.1740112
20: -61.0928268, 52.9272346, -61.1007690, 52.9410858, -114.0339050, 114.0279922
21: -84.8438187, 57.9199371, -84.8566208, 57.9333496, -142.7771606, 142.7765503
22: -85.5610504, 58.3013802, -85.6062164, 58.3128853, -143.8739319, 143.9075928
23: -70.1502151, 57.4744263, -70.1692810, 57.4879646, -127.6381836, 127.6437073
24: -79.3005676, 53.0085220, -79.3250732, 53.0148888, -132.3154449, 132.3335876
25: -72.8486633, 62.1973076, -72.8708496, 62.2103882, -135.0590515, 135.0681610
26: -98.5812836, 87.5292816, -98.5976486, 87.5505066, -186.1317902, 186.1269226
27: -85.7818527, 59.0805893, -85.8103485, 59.0910378, -144.8728943, 144.8909302
28: -69.9299545, 63.7308273, -69.9463654, 63.7397728, -133.6697235, 133.6771851
29: -88.4804688, 51.1481628, -88.5082321, 51.1629524, -139.6434174, 139.6563721
30: -87.1215744, 65.4403152, -87.1399994, 65.4606552, -152.5822296, 152.5803223
31: -85.1856079, 55.5028381, -85.2121735, 55.5119781, -140.6975861, 140.7150116
32: -75.8148193, 62.1156082, -75.8260498, 62.1426926, -137.9575195, 137.9416504
33: -108.0610046, 82.4846039, -108.0926743, 82.4938812, -190.5548859, 190.5772705
34: -87.7829437, 65.8608246, -87.8044815, 65.8734283, -153.6563721, 153.6652985
35: -83.9422913, 68.6341705, -83.9670258, 68.6460266, -152.5883026, 152.6011963
36: -82.9066315, 73.6586456, -82.9220123, 73.6702118, -156.5768433, 156.5806580
37: -123.5701141, 71.2022095, -123.6067963, 71.2105560, -194.7806702, 194.8090057
38: -101.3471451, 93.1538849, -101.3687668, 93.1758041, -194.5229492, 194.5226440
39: -115.5343170, 83.5324402, -115.5740662, 83.5430756, -199.0773773, 199.1064911
40: -98.8452148, 60.3014030, -98.8651199, 60.3090744, -159.1542664, 159.1665192
41: -78.8631668, 63.0488968, -78.8855286, 63.0587807, -141.9219513, 141.9344177
42: -63.6271095, 59.0463715, -63.6368828, 59.0803032, -122.7074127, 122.6832504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=446, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=636, inp2_unstable=636, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5717461, upper bound: 112.8623043
time: 85.46 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6869581, upper bound: 112.8623043
time: 89.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 177.20 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.5710057, upper bound: 112.5422846
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.6863782, upper bound: 112.5422846
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.5717461, upper bound: 112.6989059
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.5717461, upper bound: 112.6989059
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.5710057, upper bound: 112.5429650
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.6863782, upper bound: 112.5429650
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.5717461, upper bound: 112.6997387
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.6869581, upper bound: 112.6997387
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.5710057, upper bound: 112.7120399
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.6863782, upper bound: 112.7120399
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.5717461, upper bound: 112.8617321
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.5717461, upper bound: 112.8617321
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.5710057, upper bound: 112.7128802
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.6863782, upper bound: 112.7128802
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.5717461, upper bound: 112.8623043
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 177.20
Output dim: 12, lower bound: -112.6869581, upper bound: 112.8623043
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=177.0996856689453
rel_dist={12: [-112.90240024953403, 112.90240025266431]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

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
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2451079, upper bound: 111.1196876
time: 115.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2451079, upper bound: 111.2451077
time: 99.16 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 214.59 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 214.59
Output dim: 12, lower bound: -111.2451079, upper bound: 111.1196876
IS_A2, status: Status.UNKNOWN, split count: 1, time: 214.59
Output dim: 12, lower bound: -111.2451079, upper bound: 111.2451077

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -105.7478790, 76.8107834, -105.9741974, 76.9708557, -182.7187347, 182.7849731
1: -62.6895599, 57.5903893, -62.8271523, 57.6920547, -120.3816147, 120.4175415
2: -58.2942429, 58.0871048, -58.4947777, 58.2218666, -116.5160980, 116.5818787
3: -67.6614227, 66.6938324, -67.8558044, 66.8612213, -134.5226288, 134.5496368
4: -73.0185394, 69.1616516, -73.2631149, 69.3267288, -142.3452759, 142.4247742
5: -69.9693909, 71.5915070, -70.1697464, 71.7850647, -141.7544556, 141.7612610
6: -81.2805939, 61.1545334, -81.4757385, 61.3253365, -142.6059265, 142.6302490
7: -77.6139984, 64.8577576, -77.8369904, 64.9824829, -142.5964813, 142.6947327
8: -81.2684784, 78.5836639, -81.4839401, 78.7404480, -160.0089264, 160.0675964
9: -71.6087952, 68.1343994, -71.8049850, 68.3788605, -139.9876556, 139.9393768
10: -97.5931625, 85.9088516, -98.0306168, 86.3378143, -183.9309692, 183.9394684
11: -92.5323410, 65.1348572, -92.9090424, 65.4214630, -157.9537964, 158.0438995
12: -86.4826508, 89.0872192, -86.9466858, 89.5160904, -175.9987335, 176.0339050
13: -90.2257690, 97.9576340, -90.4438095, 98.2575607, -188.4832764, 188.4014435
14: -133.1739807, 75.6484756, -133.6940918, 76.0285492, -209.2025299, 209.3425293
15: -84.7683563, 63.9488258, -85.0461655, 64.1821594, -148.9505157, 148.9949951
16: -101.3795853, 68.7092209, -101.6476593, 68.9830780, -170.3626709, 170.3568726
17: -135.5813141, 81.3465424, -136.0321350, 81.6841049, -217.2654114, 217.3786774
18: -85.3504333, 68.8810425, -85.6380692, 69.0252991, -154.3757324, 154.5191040
19: -68.3848419, 48.4514236, -68.5614929, 48.5245590, -116.9093933, 117.0129089
20: -60.8479576, 52.7312737, -61.0339966, 52.8478813, -113.6958237, 113.7652740
21: -84.5057297, 57.6774330, -84.7726974, 57.8251228, -142.3308563, 142.4501343
22: -85.2066956, 58.0677567, -85.4511719, 58.2527275, -143.4594269, 143.5189209
23: -69.9317780, 57.3499985, -70.0969543, 57.4449615, -127.3767242, 127.4469528
24: -78.8860855, 52.8611717, -79.1230698, 52.9871941, -131.8732758, 131.9842377
25: -72.6233521, 62.0051537, -72.7747574, 62.1396713, -134.7630157, 134.7799072
26: -98.2069397, 87.1182098, -98.4993591, 87.3634033, -185.5703125, 185.6175537
27: -85.2542801, 58.8834457, -85.5526886, 59.0614166, -144.3157043, 144.4361267
28: -69.6882782, 63.5879440, -69.8435135, 63.7049103, -133.3931885, 133.4314575
29: -88.2226257, 50.9105492, -88.4153595, 51.0690575, -139.2916870, 139.3258820
30: -86.8901520, 65.1539383, -87.0632553, 65.3281937, -152.2183533, 152.2171936
31: -84.8343201, 55.4123650, -85.0784912, 55.4869728, -140.3212891, 140.4908600
32: -75.5650330, 61.7727699, -75.7444916, 61.9647942, -137.5298157, 137.5172577
33: -107.5151291, 82.1626663, -107.8181458, 82.4406128, -189.9557495, 189.9808044
34: -87.4266281, 65.5697861, -87.6335297, 65.8256149, -153.2522125, 153.2033081
35: -83.5157547, 68.3576660, -83.7514114, 68.6007385, -152.1164703, 152.1090698
36: -82.6186447, 73.5060120, -82.7996063, 73.6294098, -156.2480164, 156.3056030
37: -123.0927048, 71.0162354, -123.3877335, 71.1699448, -194.2626495, 194.4039612
38: -100.9975967, 92.9542847, -101.2305984, 93.1195374, -194.1171265, 194.1848755
39: -115.1795654, 83.3827515, -115.4268799, 83.5090790, -198.6886444, 198.8096313
40: -98.3746643, 60.0926590, -98.6586380, 60.2842369, -158.6588745, 158.7512970
41: -78.5238266, 62.8973999, -78.7318878, 63.0184097, -141.5422363, 141.6292877
42: -63.3596535, 58.6848717, -63.5603981, 58.9085655, -122.2682190, 122.2452698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=448, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=635, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

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
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -111.0962632, upper bound: 111.1053148
time: 108.88 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -111.0962632, upper bound: 111.1078272
time: 81.94 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -106.1341934, 77.0433197, -106.1503143, 77.0492172, -183.1834106, 183.1936188
1: -62.9125290, 57.7340736, -62.9232979, 57.7399750, -120.6525040, 120.6573563
2: -58.6830139, 58.2630463, -58.6953888, 58.2687340, -116.9517365, 116.9584351
3: -68.0199890, 66.9307556, -68.0358582, 66.9384003, -134.9583893, 134.9666138
4: -73.5132523, 69.3889694, -73.5328445, 69.3965607, -142.9098206, 142.9217987
5: -70.3380203, 71.8568115, -70.3547668, 71.8643341, -142.2023621, 142.2115784
6: -81.5817413, 61.4390564, -81.5939560, 61.4598351, -143.0415649, 143.0330200
7: -77.9623184, 65.0399628, -77.9781876, 65.0487900, -143.0111084, 143.0181580
8: -81.6675262, 78.8113861, -81.6838989, 78.8181229, -160.4856567, 160.4952850
9: -71.8656158, 68.6264420, -71.8742065, 68.6443481, -140.5099487, 140.5006409
10: -98.1372833, 86.8112640, -98.1497650, 86.8441467, -184.9813995, 184.9610291
11: -92.9973526, 65.7273712, -93.0064392, 65.7506714, -158.7480164, 158.7337952
12: -87.0223999, 89.9753723, -87.0318298, 90.0053787, -177.0277710, 177.0072021
13: -90.5028687, 98.5130844, -90.5143051, 98.5315933, -189.0344543, 189.0273743
14: -133.8108521, 76.4723053, -133.8265686, 76.4968567, -210.3077087, 210.2988739
15: -85.2975311, 64.2654114, -85.3187790, 64.2728500, -149.5703735, 149.5841827
16: -101.7602463, 69.2506409, -101.7733765, 69.2722626, -171.0325012, 171.0240173
17: -136.1045074, 82.0433655, -136.1165161, 82.0592422, -218.1637268, 218.1598816
18: -85.8267746, 69.0953674, -85.8461838, 69.1091156, -154.9358826, 154.9415436
19: -68.6331329, 48.5608444, -68.6513748, 48.5678635, -117.2009964, 117.2122192
20: -61.1089859, 52.9547348, -61.1150551, 52.9651489, -114.0741196, 114.0697937
21: -84.8646088, 57.9534798, -84.8740616, 57.9646759, -142.8292847, 142.8275452
22: -85.6066132, 58.3436966, -85.6327515, 58.3549957, -143.9616089, 143.9764404
23: -70.1796722, 57.4995766, -70.1890717, 57.5109749, -127.6906433, 127.6886368
24: -79.3418427, 53.0184937, -79.3577576, 53.0242310, -132.3660736, 132.3762512
25: -72.8784790, 62.2197151, -72.8910522, 62.2304192, -135.1089020, 135.1107635
26: -98.6059189, 87.5881500, -98.6173630, 87.6040192, -186.2098999, 186.2055054
27: -85.8276062, 59.0948372, -85.8465805, 59.1031799, -144.9307861, 144.9414062
28: -69.9539413, 63.7523041, -69.9629669, 63.7606506, -133.7145844, 133.7152710
29: -88.5135345, 51.1854858, -88.5305939, 51.1982880, -139.7118225, 139.7160645
30: -87.1540222, 65.4814453, -87.1628265, 65.4975128, -152.6515350, 152.6442566
31: -85.2207870, 55.5220871, -85.2379074, 55.5288620, -140.7496490, 140.7599945
32: -75.8331146, 62.1546898, -75.8424835, 62.1697845, -138.0028992, 137.9971771
33: -108.1132355, 82.5019226, -108.1358948, 82.5098190, -190.6230469, 190.6378174
34: -87.8157349, 65.8787384, -87.8322830, 65.8889771, -153.7047119, 153.7110291
35: -83.9834976, 68.6544952, -84.0028229, 68.6610413, -152.6445312, 152.6572876
36: -82.9299316, 73.6796112, -82.9456482, 73.6867523, -156.6166840, 156.6252594
37: -123.6186905, 71.2336426, -123.6416855, 71.2410278, -194.8597107, 194.8753052
38: -101.3820648, 93.1789780, -101.4007797, 93.1924438, -194.5745087, 194.5797424
39: -115.5723572, 83.5531464, -115.6012115, 83.5592957, -199.1316528, 199.1543274
40: -98.8797226, 60.3155746, -98.8938370, 60.3214722, -159.2012024, 159.2094116
41: -78.8964462, 63.0787048, -78.9103928, 63.0889168, -141.9853668, 141.9891052
42: -63.6430740, 59.0981865, -63.6516342, 59.1236115, -122.7666855, 122.7498169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=448, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=636, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

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
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1421
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1339
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1353
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1497
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1389
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1358
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1429
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1496
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1333
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1430
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1323
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1352
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1315

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.0962632, upper bound: 111.2371982
time: 93.89 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.0962632, upper bound: 111.2386723
time: 98.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 195.07 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 195.07
Output dim: 12, lower bound: -111.0962632, upper bound: 111.1053148
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 195.07
Output dim: 12, lower bound: -111.0962632, upper bound: 111.1078272
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 195.07
Output dim: 12, lower bound: -111.0962632, upper bound: 111.2371982
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 195.07
Output dim: 12, lower bound: -111.0962632, upper bound: 111.2386723

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -105.9227829, 76.9967499, -105.7186203, 76.6913986, -182.6141663, 182.7153625
1: -62.7706032, 57.7028313, -62.6375084, 57.4959068, -120.2665100, 120.3403320
2: -58.4370575, 58.2314835, -58.2410011, 57.9623795, -116.3994370, 116.4724808
3: -67.7587128, 66.8791199, -67.5559235, 66.5710831, -134.3298035, 134.4350433
4: -73.2365112, 69.3450928, -73.0144348, 69.0455475, -142.2820587, 142.3595276
5: -70.0714798, 71.8062286, -69.8550110, 71.4514313, -141.5229034, 141.6612396
6: -81.4950104, 61.3428726, -81.3371811, 61.2357407, -142.7307434, 142.6800537
7: -77.7476883, 65.0009003, -77.5317535, 64.7449646, -142.4926453, 142.5326538
8: -81.4498978, 78.7642059, -81.2629242, 78.4673157, -159.9172058, 160.0270996
9: -71.8000107, 68.4225769, -71.5852280, 68.2407532, -140.0407715, 140.0078125
10: -98.0327759, 86.3893356, -97.5480270, 86.0665283, -184.0993042, 183.9373627
11: -92.9274826, 65.4391937, -92.5537796, 65.2288055, -158.1562805, 157.9929810
12: -86.9667816, 89.5208817, -86.4039917, 89.1777344, -176.1445160, 175.9248657
13: -90.4346161, 98.3785782, -90.3378677, 98.1991653, -188.6337891, 188.7164459
14: -133.6913757, 76.1638641, -133.2735138, 75.9357300, -209.6270905, 209.4373779
15: -85.0589142, 64.1945343, -84.8486786, 63.9862480, -149.0451660, 149.0431976
16: -101.6569443, 69.0377045, -101.3825073, 68.8477249, -170.5046387, 170.4201965
17: -136.0322418, 81.7956085, -135.6556549, 81.5752716, -217.6075134, 217.4512634
18: -85.6938858, 68.9885483, -85.4591446, 68.8747864, -154.5686646, 154.4476929
19: -68.5643463, 48.4748344, -68.3633881, 48.3973465, -116.9616928, 116.8382263
20: -61.0439110, 52.8415184, -60.8468018, 52.7494621, -113.7933655, 113.6883163
21: -84.7958679, 57.7801437, -84.4879227, 57.6357307, -142.4315948, 142.2680664
22: -85.5286407, 58.2102585, -85.3943253, 58.0519257, -143.5805664, 143.6045837
23: -70.1190872, 57.4034081, -69.9472961, 57.3160782, -127.4351654, 127.3507080
24: -79.2342606, 52.9847946, -79.1077194, 52.9357300, -132.1699677, 132.0925140
25: -72.8213959, 62.1040764, -72.7092361, 61.9843521, -134.8057556, 134.8133087
26: -98.5243530, 87.2598114, -98.1134262, 86.9946365, -185.5189819, 185.3732300
27: -85.6752548, 59.0542145, -85.5051193, 58.9680328, -144.6432800, 144.5593262
28: -69.8943863, 63.6939316, -69.7603683, 63.6206779, -133.5150604, 133.4542999
29: -88.4524002, 51.0166893, -88.2985840, 50.8708687, -139.3232727, 139.3152618
30: -87.0956039, 65.3013153, -86.8796005, 65.1519470, -152.2475586, 152.1809082
31: -85.1188202, 55.4547653, -84.8948746, 55.3812447, -140.5000610, 140.3496399
32: -75.7678528, 61.9980164, -75.5779419, 61.8709450, -137.6387939, 137.5759430
33: -107.9062347, 82.4350586, -107.7241669, 82.2080536, -190.1142578, 190.1592255
34: -87.6791306, 65.8174820, -87.5378342, 65.6385040, -153.3176270, 153.3553162
35: -83.8127594, 68.6030884, -83.6734467, 68.4112015, -152.2239685, 152.2765350
36: -82.8399963, 73.6264801, -82.7306061, 73.5272675, -156.3672638, 156.3570862
37: -123.5096436, 71.1378326, -123.3394089, 71.0119247, -194.5215759, 194.4772339
38: -101.2287827, 93.1265869, -101.0483780, 92.9385681, -194.1673431, 194.1749573
39: -115.4586563, 83.5007553, -115.2935791, 83.3717194, -198.8303680, 198.7943268
40: -98.7669678, 60.2817230, -98.5915222, 60.1598511, -158.9268188, 158.8732452
41: -78.8040314, 63.0104599, -78.6804504, 62.9134216, -141.7174530, 141.6909027
42: -63.5803299, 58.8884926, -63.3708725, 58.7151604, -122.2954865, 122.2593689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=636, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -111.0778331, upper bound: 111.0952547
time: 88.53 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 12, lower bound: -111.0852586, upper bound: 111.2253961
time: 95.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -106.1145859, 77.0388184, -106.1187210, 77.0420837, -183.1566772, 183.1575317
1: -62.8982010, 57.7309723, -62.9002342, 57.7351379, -120.6333389, 120.6312027
2: -58.6673317, 58.2593422, -58.6696281, 58.2629242, -116.9302521, 116.9289703
3: -68.0012817, 66.9244843, -68.0046539, 66.9284210, -134.9297028, 134.9291382
4: -73.4907074, 69.3831635, -73.4965973, 69.3875122, -142.8782196, 142.8797607
5: -70.3201294, 71.8510437, -70.3263550, 71.8550568, -142.1751862, 142.1773987
6: -81.5713348, 61.4157104, -81.5775146, 61.4201050, -142.9914246, 142.9932251
7: -77.9420471, 65.0355377, -77.9480667, 65.0418091, -142.9838562, 142.9836121
8: -81.6502228, 78.8059464, -81.6558762, 78.8094559, -160.4596558, 160.4618225
9: -71.8583984, 68.6080933, -71.8627701, 68.6147385, -140.4731445, 140.4708557
10: -98.1263885, 86.7743912, -98.1323853, 86.7844849, -184.9108734, 184.9067688
11: -92.9887009, 65.7031250, -92.9926453, 65.7116852, -158.7003784, 158.6957703
12: -87.0157394, 89.9433517, -87.0213776, 89.9503098, -176.9660492, 176.9647217
13: -90.4934235, 98.4950943, -90.4997940, 98.5019531, -188.9953766, 188.9948730
14: -133.7977600, 76.4545441, -133.8058319, 76.4629822, -210.2607422, 210.2603760
15: -85.2756805, 64.2566452, -85.2842941, 64.2588501, -149.5345306, 149.5409241
16: -101.7483368, 69.2303925, -101.7542038, 69.2401581, -170.9884949, 170.9845886
17: -136.0964966, 82.0265427, -136.1040039, 82.0320587, -218.1285553, 218.1305542
18: -85.8112030, 69.0816956, -85.8209381, 69.0875092, -154.8987122, 154.9026337
19: -68.6260681, 48.5537643, -68.6401062, 48.5567589, -117.1828308, 117.1938629
20: -61.1023483, 52.9440117, -61.1045303, 52.9478073, -114.0501556, 114.0485382
21: -84.8566055, 57.9387856, -84.8613968, 57.9410782, -142.7976685, 142.8001709
22: -85.5978622, 58.3244781, -85.6189423, 58.3236847, -143.9215393, 143.9434052
23: -70.1732559, 57.4885139, -70.1787643, 57.4933014, -127.6665497, 127.6672745
24: -79.3297424, 53.0140839, -79.3366470, 53.0174332, -132.3471680, 132.3507385
25: -72.8715973, 62.2101669, -72.8800507, 62.2153130, -135.0868988, 135.0902100
26: -98.5971222, 87.5654907, -98.6035690, 87.5674591, -186.1645813, 186.1690369
27: -85.8126221, 59.0888901, -85.8222351, 59.0939255, -144.9065552, 144.9111328
28: -69.9481277, 63.7418289, -69.9536133, 63.7442207, -133.6923523, 133.6954346
29: -88.5050201, 51.1675835, -88.5172653, 51.1712074, -139.6762238, 139.6848450
30: -87.1465225, 65.4658661, -87.1509094, 65.4727173, -152.6192322, 152.6167755
31: -85.2107468, 55.5144119, -85.2217407, 55.5169868, -140.7277222, 140.7361450
32: -75.8253021, 62.1466560, -75.8300781, 62.1555405, -137.9808350, 137.9767303
33: -108.0943985, 82.4948578, -108.1053238, 82.4985657, -190.5929565, 190.6001892
34: -87.8028259, 65.8715286, -87.8114319, 65.8776855, -153.6804810, 153.6829529
35: -83.9668808, 68.6494598, -83.9759216, 68.6528854, -152.6197662, 152.6253815
36: -82.9167786, 73.6733856, -82.9244843, 73.6767120, -156.5934753, 156.5978699
37: -123.6053848, 71.2198181, -123.6205139, 71.2189331, -194.8243103, 194.8403320
38: -101.3656616, 93.1728516, -101.3750229, 93.1825027, -194.5481567, 194.5478516
39: -115.5595169, 83.5473633, -115.5806885, 83.5500565, -199.1095581, 199.1280518
40: -98.8674088, 60.3102684, -98.8739471, 60.3130226, -159.1804352, 159.1842194
41: -78.8866959, 63.0636444, -78.8949051, 63.0648613, -141.9515381, 141.9585571
42: -63.6357536, 59.0760269, -63.6400528, 59.0900536, -122.7258072, 122.7160797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=447, inp2_unstable=447, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=636, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1421
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1339
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1353
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1497
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1389
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1358
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1429
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1496
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1333
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1430
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1323
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1352
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1315

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -111.0778331, upper bound: 111.0977241
time: 98.87 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 12, lower bound: -111.0852586, upper bound: 111.2269127
time: 100.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 201.84 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 201.84
Output dim: 12, lower bound: -111.0778331, upper bound: 111.0952547
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 201.84
Output dim: 12, lower bound: -111.0852586, upper bound: 111.2253961
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 201.84
Output dim: 12, lower bound: -111.0778331, upper bound: 111.0977241
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 201.84
Output dim: 12, lower bound: -111.0852586, upper bound: 111.2269127
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=177.0996856689453
rel_dist={12: [-111.25869176025041, 111.25869175221396]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 9578.94 seconds
