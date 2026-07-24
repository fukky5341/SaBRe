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
execution time: IAR + LP analysis = 2.76 + 104.41 = 107.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -120.7499906, upper bound: 120.7499906


# Binary Search by BASE starts (time budget: 17892.83 seconds, max iter: 100)

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
Binary search time: 496.19 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 17396.64 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

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
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8804127, upper bound: 116.6609588
time: 91.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6609588, upper bound: 116.8804127
time: 81.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 173.04 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 173.04
Output dim: 12, lower bound: -116.8804127, upper bound: 116.6609588
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 173.04
Output dim: 12, lower bound: -116.6609588, upper bound: 116.8804127

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8679768, upper bound: 116.4866605
time: 89.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6266655, upper bound: 116.6363803
time: 71.92 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6363803, upper bound: 116.6266655
time: 80.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.4866605, upper bound: 116.8679769
time: 88.99 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 172.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 172.25
Output dim: 12, lower bound: -116.8679768, upper bound: 116.4866605
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 172.25
Output dim: 12, lower bound: -116.6266655, upper bound: 116.6363803
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 172.25
Output dim: 12, lower bound: -116.6363803, upper bound: 116.6266655
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 172.25
Output dim: 12, lower bound: -116.4866605, upper bound: 116.8679769

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8413663, upper bound: 116.2536510
time: 117.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6390475, upper bound: 116.4667786
time: 1089.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6030408, upper bound: 116.4073181
time: 100.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.3957490, upper bound: 116.6138091
time: 87.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6138091, upper bound: 116.3957490
time: 85.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.4073180, upper bound: 116.6030408
time: 111.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.4667786, upper bound: 116.6390475
time: 83.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.2536510, upper bound: 116.8413663
time: 215.03 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 300.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 300.43
Output dim: 12, lower bound: -116.8413663, upper bound: 116.2536510
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 300.43
Output dim: 12, lower bound: -116.6390475, upper bound: 116.4667786
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 300.43
Output dim: 12, lower bound: -116.6030408, upper bound: 116.4073181
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 300.43
Output dim: 12, lower bound: -116.3957490, upper bound: 116.6138091
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 300.43
Output dim: 12, lower bound: -116.6138091, upper bound: 116.3957490
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 300.43
Output dim: 12, lower bound: -116.4073180, upper bound: 116.6030408
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 300.43
Output dim: 12, lower bound: -116.4667786, upper bound: 116.6390475
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 300.43
Output dim: 12, lower bound: -116.2536510, upper bound: 116.8413663

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8274637, upper bound: 116.1525468
time: 104.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.7087830, upper bound: 116.2312956
time: 81.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.6255279, upper bound: 116.3677142
time: 84.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.5040150, upper bound: 116.4450104
time: 112.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.5796625, upper bound: 116.2639456
time: 109.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.5251364, upper bound: 116.3928017
time: 73.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.3721171, upper bound: 116.4737059
time: 82.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.3170625, upper bound: 116.5992722
time: 80.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.5992722, upper bound: 116.3170625
time: 1197.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.4737059, upper bound: 116.3721171
time: 114.19 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1313.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1313.71
Output dim: 12, lower bound: -116.8274637, upper bound: 116.1525468
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1313.71
Output dim: 12, lower bound: -116.7087830, upper bound: 116.2312956
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1313.71
Output dim: 12, lower bound: -116.6255279, upper bound: 116.3677142
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1313.71
Output dim: 12, lower bound: -116.5040150, upper bound: 116.4450104
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1313.71
Output dim: 12, lower bound: -116.5796625, upper bound: 116.2639456
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1313.71
Output dim: 12, lower bound: -116.5251364, upper bound: 116.3928017
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1313.71
Output dim: 12, lower bound: -116.3721171, upper bound: 116.4737059
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1313.71
Output dim: 12, lower bound: -116.3170625, upper bound: 116.5992722
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1313.71
Output dim: 12, lower bound: -116.5992722, upper bound: 116.3170625
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1313.71
Output dim: 12, lower bound: -116.4737059, upper bound: 116.3721171
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1313.71
Output dim: 12, lower bound: -116.4073180, upper bound: 116.6030408
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 1313.71
Output dim: 12, lower bound: -116.4667786, upper bound: 116.6390475
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 1313.71
Output dim: 12, lower bound: -116.2536510, upper bound: 116.8413663
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=177.0996856689453
rel_dist={12: [-116.89836616746769, 116.89836617699747]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

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
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8894997, upper bound: 112.7313450
time: 1328.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.7313450, upper bound: 112.8894997
time: 90.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1419.04 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1419.04
Output dim: 12, lower bound: -112.8894997, upper bound: 112.7313450
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1419.04
Output dim: 12, lower bound: -112.7313450, upper bound: 112.8894997

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8837386, upper bound: 112.6120021
time: 82.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.7098969, upper bound: 112.7185355
time: 86.22 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.7185355, upper bound: 112.7098969
time: 94.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6120021, upper bound: 112.8837386
time: 105.54 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 202.63 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 202.63
Output dim: 12, lower bound: -112.8837386, upper bound: 112.6120021
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 202.63
Output dim: 12, lower bound: -112.7098969, upper bound: 112.7185355
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 202.63
Output dim: 12, lower bound: -112.7185355, upper bound: 112.7098969
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 202.63
Output dim: 12, lower bound: -112.6120021, upper bound: 112.8837386

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8730512, upper bound: 112.4477191
time: 96.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.7239083, upper bound: 112.6019225
time: 79.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6998107, upper bound: 112.5569277
time: 83.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5484925, upper bound: 112.7089610
time: 282.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.7089610, upper bound: 112.5484925
time: 87.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5569277, upper bound: 112.6998107
time: 94.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6019225, upper bound: 112.7239083
time: 113.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.4477191, upper bound: 112.8730512
time: 107.12 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 223.38 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 223.38
Output dim: 12, lower bound: -112.8730512, upper bound: 112.4477191
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 223.38
Output dim: 12, lower bound: -112.7239083, upper bound: 112.6019225
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 223.38
Output dim: 12, lower bound: -112.6998107, upper bound: 112.5569277
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 223.38
Output dim: 12, lower bound: -112.5484925, upper bound: 112.7089610
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 223.38
Output dim: 12, lower bound: -112.7089610, upper bound: 112.5484925
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 223.38
Output dim: 12, lower bound: -112.5569277, upper bound: 112.6998107
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 223.38
Output dim: 12, lower bound: -112.6019225, upper bound: 112.7239083
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 223.38
Output dim: 12, lower bound: -112.4477191, upper bound: 112.8730512

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8612117, upper bound: 112.3698843
time: 84.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.7653048, upper bound: 112.4311734
time: 82.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.7121477, upper bound: 112.5218686
time: 86.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6146154, upper bound: 112.5854814
time: 87.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6801666, upper bound: 112.4450887
time: 118.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6356681, upper bound: 112.5450574
time: 90.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5288144, upper bound: 112.5984575
time: 104.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.4835316, upper bound: 112.6972459
time: 110.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.6972458, upper bound: 112.4835316
time: 84.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5984575, upper bound: 112.5288144
time: 100.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.5450574, upper bound: 112.6356681
time: 107.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.4450887, upper bound: 112.6801666
time: 112.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 222.62 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.62
Output dim: 12, lower bound: -112.8612117, upper bound: 112.3698843
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.62
Output dim: 12, lower bound: -112.7653048, upper bound: 112.4311734
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.62
Output dim: 12, lower bound: -112.7121477, upper bound: 112.5218686
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.62
Output dim: 12, lower bound: -112.6146154, upper bound: 112.5854814
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.62
Output dim: 12, lower bound: -112.6801666, upper bound: 112.4450887
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.62
Output dim: 12, lower bound: -112.6356681, upper bound: 112.5450574
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.62
Output dim: 12, lower bound: -112.5288144, upper bound: 112.5984575
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.62
Output dim: 12, lower bound: -112.4835316, upper bound: 112.6972459
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.62
Output dim: 12, lower bound: -112.6972458, upper bound: 112.4835316
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.62
Output dim: 12, lower bound: -112.5984575, upper bound: 112.5288144
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.62
Output dim: 12, lower bound: -112.5450574, upper bound: 112.6356681
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.62
Output dim: 12, lower bound: -112.4450887, upper bound: 112.6801666
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 222.62
Output dim: 12, lower bound: -112.6019225, upper bound: 112.7239083
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 222.62
Output dim: 12, lower bound: -112.4477191, upper bound: 112.8730512
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=177.0996856689453
rel_dist={12: [-112.90240024953403, 112.90240025266431]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

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
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1687

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2451079, upper bound: 111.1196876
time: 93.74 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.1196876, upper bound: 111.2451079
time: 91.63 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 185.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 185.52
Output dim: 12, lower bound: -111.2451079, upper bound: 111.1196876
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 185.52
Output dim: 12, lower bound: -111.1196876, upper bound: 111.2451079

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2386724, upper bound: 111.0229757
time: 2125.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -111.0999556, upper bound: 111.1078272
time: 90.85 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -111.1078272, upper bound: 111.0999556
time: 132.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.0229757, upper bound: 111.2386724
time: 160.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 295.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 295.04
Output dim: 12, lower bound: -111.2386724, upper bound: 111.0229757
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 295.04
Output dim: 12, lower bound: -111.0999556, upper bound: 111.1078272
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 295.04
Output dim: 12, lower bound: -111.1078272, upper bound: 111.0999556
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 295.04
Output dim: 12, lower bound: -111.0229757, upper bound: 111.2386724

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -111.2269132, upper bound: 110.8912853
time: 94.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -111.1101967, upper bound: 111.0124660
time: 111.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -106.1784668, 77.0588989, -106.1784668, 77.0588989, -183.2373352, 183.2373505
1: -62.9409370, 57.7498055, -62.9409370, 57.7498055, -120.6907272, 120.6907425
2: -58.7195435, 58.2779427, -58.7195435, 58.2779427, -116.9974823, 116.9974823
3: -68.0662384, 66.9507904, -68.0662384, 66.9507904, -135.0170288, 135.0170288
4: -73.5633163, 69.4090347, -73.5633163, 69.4090347, -142.9723511, 142.9723358
5: -70.3835831, 71.8765717, -70.3835831, 71.8765717, -142.2601471, 142.2601624
6: -81.6132431, 61.4956169, -81.6132431, 61.4956169, -143.1088562, 143.1088562
7: -78.0063248, 65.0635223, -78.0063248, 65.0635223, -143.0698395, 143.0698395
8: -81.7095795, 78.8292465, -81.7095795, 78.8292465, -160.5388184, 160.5388184
9: -71.8883057, 68.6721191, -71.8883057, 68.6721191, -140.5604248, 140.5604248
10: -98.1695099, 86.8949890, -98.1695099, 86.8949890, -185.0644989, 185.0644989
11: -93.0212784, 65.7872467, -93.0212784, 65.7872467, -158.8085022, 158.8085175
12: -87.0472412, 90.0524597, -87.0472412, 90.0524597, -177.0996857, 177.0996857
13: -90.5337372, 98.5605316, -90.5337372, 98.5605316, -189.0942688, 189.0942535
14: -133.8533020, 76.5364532, -133.8533020, 76.5364532, -210.3897552, 210.3897552
15: -85.3527527, 64.2846985, -85.3527527, 64.2846985, -149.6374359, 149.6374512
16: -101.7953720, 69.3070221, -101.7953720, 69.3070221, -171.1023865, 171.1023865
17: -136.1369781, 82.0875778, -136.1369781, 82.0875778, -218.2244873, 218.2245178
18: -85.8764343, 69.1314926, -85.8764343, 69.1314926, -155.0079346, 155.0079041
19: -68.6804276, 48.5791245, -68.6804276, 48.5791245, -117.2595367, 117.2595367
20: -61.1254463, 52.9818954, -61.1254463, 52.9818954, -114.1073380, 114.1073456
21: -84.8898087, 57.9831657, -84.8898087, 57.9831657, -142.8729706, 142.8729706
22: -85.6746979, 58.3725815, -85.6746979, 58.3725815, -144.0472717, 144.0472717
23: -70.2044983, 57.5289650, -70.2044983, 57.5289650, -127.7334595, 127.7334595
24: -79.3837662, 53.0343819, -79.3837662, 53.0343819, -132.4181519, 132.4181519
25: -72.9129944, 62.2476158, -72.9129944, 62.2476158, -135.1606140, 135.1606140
26: -98.6358109, 87.6322632, -98.6358109, 87.6322632, -186.2680664, 186.2680664
27: -85.8767395, 59.1169243, -85.8767395, 59.1169243, -144.9936523, 144.9936523
28: -69.9777527, 63.7741852, -69.9777527, 63.7741852, -133.7519379, 133.7519379
29: -88.5594101, 51.2195778, -88.5594101, 51.2195778, -139.7789917, 139.7789917
30: -87.1779556, 65.5240784, -87.1779556, 65.5240784, -152.7020264, 152.7020264
31: -85.2681046, 55.5402184, -85.2681046, 55.5402184, -140.8083191, 140.8083191
32: -75.8581848, 62.1948586, -75.8581848, 62.1948586, -138.0530396, 138.0530396
33: -108.1714706, 82.5236893, -108.1714706, 82.5236893, -190.6951599, 190.6951599
34: -87.8582535, 65.9068375, -87.8582535, 65.9068375, -153.7650909, 153.7650909
35: -84.0335083, 68.6720810, -84.0335083, 68.6720810, -152.7055969, 152.7055664
36: -82.9705963, 73.6981812, -82.9705963, 73.6981812, -156.6687775, 156.6687775
37: -123.6785660, 71.2537994, -123.6785660, 71.2537994, -194.9323730, 194.9323730
38: -101.4305344, 93.2145233, -101.4305344, 93.2145233, -194.6450500, 194.6450500
39: -115.6468658, 83.5707703, -115.6468658, 83.5707703, -199.2175903, 199.2176056
40: -98.9180450, 60.3319740, -98.9180450, 60.3319740, -159.2500153, 159.2500000
41: -78.9323273, 63.1058197, -78.9323273, 63.1058197, -142.0381470, 142.0381470
42: -63.6651802, 59.1662407, -63.6651802, 59.1662407, -122.8314209, 122.8314209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1341

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1688

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 12, lower bound: -111.0124660, upper bound: 111.1101967
time: 101.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -110.8912853, upper bound: 111.2269132
time: 112.01 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 215.68 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 215.68
Output dim: 12, lower bound: -111.2269132, upper bound: 110.8912853
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 215.68
Output dim: 12, lower bound: -111.1101967, upper bound: 111.0124660
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 215.68
Output dim: 12, lower bound: -111.0124660, upper bound: 111.1101967
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 215.68
Output dim: 12, lower bound: -110.8912853, upper bound: 111.2269132
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=177.0996856689453
rel_dist={12: [-111.25869176025041, 111.25869175221396]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 11889.81 seconds
