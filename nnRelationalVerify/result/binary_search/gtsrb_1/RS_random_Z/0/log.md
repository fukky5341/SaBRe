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
execution time: IAR + LP analysis = 2.84 + 108.49 = 111.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -120.7499906, upper bound: 120.7499906


# Binary Search by BASE starts (time budget: 17888.67 seconds, max iter: 100)

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
Binary search time: 512.50 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 17376.16 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 826

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1562

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8974661, upper bound: 116.8972229
time: 249.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8972229, upper bound: 116.8974661
time: 104.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 354.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 354.38
Output dim: 12, lower bound: -116.8974661, upper bound: 116.8972229
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 354.38
Output dim: 12, lower bound: -116.8972229, upper bound: 116.8974661

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

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 922

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 861

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8973577, upper bound: 116.8939221
time: 119.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8941847, upper bound: 116.8971150
time: 114.27 seconds

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

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1656

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1542

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8799744, upper bound: 116.8795686
time: 86.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8793052, upper bound: 116.8802378
time: 120.04 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 208.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 208.59
Output dim: 12, lower bound: -116.8973577, upper bound: 116.8939221
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 208.59
Output dim: 12, lower bound: -116.8941847, upper bound: 116.8971150
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 208.59
Output dim: 12, lower bound: -116.8799744, upper bound: 116.8795686
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 208.59
Output dim: 12, lower bound: -116.8793052, upper bound: 116.8802378

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

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1781

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1694

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8624683, upper bound: 116.8918803
time: 83.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8953292, upper bound: 116.8593141
time: 71.23 seconds

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

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1547

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1346

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8936603, upper bound: 116.8885121
time: 114.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8856157, upper bound: 116.8965909
time: 100.87 seconds

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

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1692

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1541

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8736565, upper bound: 116.8703425
time: 95.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8707492, upper bound: 116.8732487
time: 82.99 seconds

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

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1432

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1782

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8439605, upper bound: 116.8780703
time: 152.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8439605, upper bound: 116.8449251
time: 100.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 255.63 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 255.63
Output dim: 12, lower bound: -116.8624683, upper bound: 116.8918803
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 255.63
Output dim: 12, lower bound: -116.8953292, upper bound: 116.8593141
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 255.63
Output dim: 12, lower bound: -116.8936603, upper bound: 116.8885121
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 255.63
Output dim: 12, lower bound: -116.8856157, upper bound: 116.8965909
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 255.63
Output dim: 12, lower bound: -116.8736565, upper bound: 116.8703425
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 255.63
Output dim: 12, lower bound: -116.8707492, upper bound: 116.8732487
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 255.63
Output dim: 12, lower bound: -116.8439605, upper bound: 116.8780703
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 255.63
Output dim: 12, lower bound: -116.8439605, upper bound: 116.8449251

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

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1567

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1587

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8182702, upper bound: 116.8498135
time: 88.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8212921, upper bound: 116.8498135
time: 77.30 seconds

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

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1740

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1734

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8406372, upper bound: 116.8450470
time: 67.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8806228, upper bound: 116.8045393
time: 167.44 seconds

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

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 906

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1397

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8921156, upper bound: 116.8698043
time: 93.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8750129, upper bound: 116.8869770
time: 98.08 seconds

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
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1387

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8758939, upper bound: 116.8943958
time: 136.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8832587, upper bound: 116.8865649
time: 88.04 seconds

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

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1775

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1400

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8705147, upper bound: 116.8253054
time: 85.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8284986, upper bound: 116.8672370
time: 166.94 seconds

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

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1004

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8706651, upper bound: 116.8629317
time: 185.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8604026, upper bound: 116.8731694
time: 91.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1313

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1707

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8276758, upper bound: 116.8672874
time: 97.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8331010, upper bound: 116.8619548
time: 98.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1318

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8739090, upper bound: 116.8438995
time: 115.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8429352, upper bound: 116.8417073
time: 104.92 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 222.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8182702, upper bound: 116.8498135
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8212921, upper bound: 116.8498135
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8406372, upper bound: 116.8450470
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8806228, upper bound: 116.8045393
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8921156, upper bound: 116.8698043
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8750129, upper bound: 116.8869770
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8758939, upper bound: 116.8943958
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8832587, upper bound: 116.8865649
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8705147, upper bound: 116.8253054
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8284986, upper bound: 116.8672370
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8706651, upper bound: 116.8629317
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8604026, upper bound: 116.8731694
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8276758, upper bound: 116.8672874
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8331010, upper bound: 116.8619548
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8739090, upper bound: 116.8438995
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 222.23
Output dim: 12, lower bound: -116.8429352, upper bound: 116.8417073

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1729

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 774

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8048431, upper bound: 116.8399935
time: 145.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8115431, upper bound: 116.8333805
time: 121.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1573

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1462

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.7975733, upper bound: 116.8495630
time: 79.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8210388, upper bound: 116.8261758
time: 78.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1385

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 906

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.7817058, upper bound: 116.8445315
time: 102.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -116.8401177, upper bound: 116.7864259
time: 83.51 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 188.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 188.31
Output dim: 12, lower bound: -116.8048431, upper bound: 116.8399935
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 188.31
Output dim: 12, lower bound: -116.8115431, upper bound: 116.8333805
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 188.31
Output dim: 12, lower bound: -116.7975733, upper bound: 116.8495630
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 188.31
Output dim: 12, lower bound: -116.8210388, upper bound: 116.8261758
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 188.31
Output dim: 12, lower bound: -116.7817058, upper bound: 116.8445315
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 188.31
Output dim: 12, lower bound: -116.8401177, upper bound: 116.7864259
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 188.31
Output dim: 12, lower bound: -116.8806228, upper bound: 116.8045393
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 188.31
Output dim: 12, lower bound: -116.8921156, upper bound: 116.8698043
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 188.31
Output dim: 12, lower bound: -116.8750129, upper bound: 116.8869770
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 188.31
Output dim: 12, lower bound: -116.8758939, upper bound: 116.8943958
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 188.31
Output dim: 12, lower bound: -116.8832587, upper bound: 116.8865649
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 188.31
Output dim: 12, lower bound: -116.8705147, upper bound: 116.8253054
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 188.31
Output dim: 12, lower bound: -116.8284986, upper bound: 116.8672370
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 188.31
Output dim: 12, lower bound: -116.8706651, upper bound: 116.8629317
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 188.31
Output dim: 12, lower bound: -116.8604026, upper bound: 116.8731694
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 188.31
Output dim: 12, lower bound: -116.8276758, upper bound: 116.8672874
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 188.31
Output dim: 12, lower bound: -116.8331010, upper bound: 116.8619548
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 188.31
Output dim: 12, lower bound: -116.8739090, upper bound: 116.8438995
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 188.31
Output dim: 12, lower bound: -116.8429352, upper bound: 116.8417073
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=177.0996856689453
rel_dist={12: [-116.89836616746769, 116.89836617699747]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1707

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8879309, upper bound: 112.8954425
time: 108.21 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8954425, upper bound: 112.8879309
time: 93.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 201.64 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 201.64
Output dim: 12, lower bound: -112.8879309, upper bound: 112.8954425
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 201.64
Output dim: 12, lower bound: -112.8954425, upper bound: 112.8879309

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

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1557

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1752

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8732407, upper bound: 112.8946013
time: 105.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8870816, upper bound: 112.8807573
time: 98.47 seconds

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
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 782

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 843

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8948722, upper bound: 112.8697802
time: 88.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8773040, upper bound: 112.8873545
time: 76.48 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 167.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 167.15
Output dim: 12, lower bound: -112.8732407, upper bound: 112.8946013
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 167.15
Output dim: 12, lower bound: -112.8870816, upper bound: 112.8807573
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 167.15
Output dim: 12, lower bound: -112.8948722, upper bound: 112.8697802
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 167.15
Output dim: 12, lower bound: -112.8773040, upper bound: 112.8873545

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
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1015

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8731040, upper bound: 112.8702422
time: 128.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8488079, upper bound: 112.8944647
time: 114.83 seconds

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

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 912

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 768

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8856397, upper bound: 112.8793059
time: 81.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8856283, upper bound: 112.8793114
time: 113.11 seconds

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

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1623

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1306

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8948121, upper bound: 112.8683923
time: 91.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8934864, upper bound: 112.8697202
time: 108.37 seconds

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

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1649

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1544

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8757182, upper bound: 112.8733567
time: 93.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8632955, upper bound: 112.8857692
time: 104.96 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 200.31 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 200.31
Output dim: 12, lower bound: -112.8731040, upper bound: 112.8702422
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 200.31
Output dim: 12, lower bound: -112.8488079, upper bound: 112.8944647
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 200.31
Output dim: 12, lower bound: -112.8856397, upper bound: 112.8793059
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 200.31
Output dim: 12, lower bound: -112.8856283, upper bound: 112.8793114
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 200.31
Output dim: 12, lower bound: -112.8948121, upper bound: 112.8683923
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 200.31
Output dim: 12, lower bound: -112.8934864, upper bound: 112.8697202
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 200.31
Output dim: 12, lower bound: -112.8757182, upper bound: 112.8733567
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 200.31
Output dim: 12, lower bound: -112.8632955, upper bound: 112.8857692

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

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1559

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8567778, upper bound: 112.8700668
time: 166.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8729310, upper bound: 112.8538837
time: 85.77 seconds

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

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 770

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1694

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8248532, upper bound: 112.8929472
time: 114.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8472630, upper bound: 112.8704671
time: 81.51 seconds

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

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1438

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 944

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8758838, upper bound: 112.8792158
time: 112.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8855495, upper bound: 112.8695720
time: 82.40 seconds

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

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1564

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8760662, upper bound: 112.8781730
time: 83.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8844933, upper bound: 112.8697595
time: 72.05 seconds

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

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1614

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1674

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8900968, upper bound: 112.8249563
time: 90.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8511790, upper bound: 112.8637447
time: 706.56 seconds

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

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1789

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8925208, upper bound: 112.8636479
time: 138.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8874201, upper bound: 112.8687548
time: 109.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 782

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 973

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8730904, upper bound: 112.8731771
time: 135.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8755352, upper bound: 112.8707724
time: 122.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1785

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 895

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8629146, upper bound: 112.8806644
time: 103.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8582067, upper bound: 112.8853879
time: 81.80 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 187.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8567778, upper bound: 112.8700668
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8729310, upper bound: 112.8538837
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8248532, upper bound: 112.8929472
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8472630, upper bound: 112.8704671
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8758838, upper bound: 112.8792158
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8855495, upper bound: 112.8695720
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8760662, upper bound: 112.8781730
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8844933, upper bound: 112.8697595
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8900968, upper bound: 112.8249563
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8511790, upper bound: 112.8637447
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8925208, upper bound: 112.8636479
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8874201, upper bound: 112.8687548
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8730904, upper bound: 112.8731771
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8755352, upper bound: 112.8707724
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8629146, upper bound: 112.8806644
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 187.28
Output dim: 12, lower bound: -112.8582067, upper bound: 112.8853879

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=448, inp2_unstable=448, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=637, inp2_unstable=637, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1438

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 905

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.7559728, upper bound: 112.8693722
time: 83.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -112.8560811, upper bound: 112.7691286
time: 84.14 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 169.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 169.83
Output dim: 12, lower bound: -112.7559728, upper bound: 112.8693722
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 169.83
Output dim: 12, lower bound: -112.8560811, upper bound: 112.7691286
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8729310, upper bound: 112.8538837
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8248532, upper bound: 112.8929472
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8472630, upper bound: 112.8704671
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8758838, upper bound: 112.8792158
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8855495, upper bound: 112.8695720
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8760662, upper bound: 112.8781730
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8844933, upper bound: 112.8697595
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8900968, upper bound: 112.8249563
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8511790, upper bound: 112.8637447
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8925208, upper bound: 112.8636479
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8874201, upper bound: 112.8687548
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8730904, upper bound: 112.8731771
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8755352, upper bound: 112.8707724
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8629146, upper bound: 112.8806644
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 169.83
Output dim: 12, lower bound: -112.8582067, upper bound: 112.8853879
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=177.0996856689453
rel_dist={12: [-112.90240024953403, 112.90240025266431]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1791

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 801

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2586866, upper bound: 111.2586796
time: 97.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2586796, upper bound: 111.2586866
time: 96.38 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 193.79 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 193.79
Output dim: 12, lower bound: -111.2586866, upper bound: 111.2586796
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 193.79
Output dim: 12, lower bound: -111.2586796, upper bound: 111.2586866

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

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1591

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1723

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2249367, upper bound: 111.2495280
time: 101.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2495351, upper bound: 111.2249295
time: 150.96 seconds

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
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1613

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1463

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2499561, upper bound: 111.2584200
time: 105.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2584131, upper bound: 111.2499632
time: 114.82 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 222.82 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 222.82
Output dim: 12, lower bound: -111.2249367, upper bound: 111.2495280
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 222.82
Output dim: 12, lower bound: -111.2495351, upper bound: 111.2249295
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 222.82
Output dim: 12, lower bound: -111.2499561, upper bound: 111.2584200
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 222.82
Output dim: 12, lower bound: -111.2584131, upper bound: 111.2499632

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
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1756

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1638

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2223852, upper bound: 111.2475686
time: 78.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2229212, upper bound: 111.2470339
time: 99.16 seconds

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

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1645

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2381718, upper bound: 111.2242543
time: 93.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2488378, upper bound: 111.2136044
time: 80.51 seconds

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

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1563

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1547

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2497266, upper bound: 111.2581515
time: 81.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2496928, upper bound: 111.2581834
time: 83.67 seconds

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

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1697

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1371

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2500472, upper bound: 111.2497319
time: 114.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2581812, upper bound: 111.2415912
time: 653.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 770.07 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 770.07
Output dim: 12, lower bound: -111.2223852, upper bound: 111.2475686
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 770.07
Output dim: 12, lower bound: -111.2229212, upper bound: 111.2470339
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 770.07
Output dim: 12, lower bound: -111.2381718, upper bound: 111.2242543
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 770.07
Output dim: 12, lower bound: -111.2488378, upper bound: 111.2136044
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 770.07
Output dim: 12, lower bound: -111.2497266, upper bound: 111.2581515
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 770.07
Output dim: 12, lower bound: -111.2496928, upper bound: 111.2581834
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 770.07
Output dim: 12, lower bound: -111.2500472, upper bound: 111.2497319
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 770.07
Output dim: 12, lower bound: -111.2581812, upper bound: 111.2415912

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

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1675

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 944

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2151369, upper bound: 111.2475110
time: 99.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2223283, upper bound: 111.2403048
time: 95.91 seconds

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

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2227553, upper bound: 111.2354828
time: 152.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2113857, upper bound: 111.2468656
time: 87.48 seconds

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
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1467

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1004

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2380581, upper bound: 111.2183929
time: 96.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2323131, upper bound: 111.2241431
time: 171.74 seconds

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
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 997

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1670

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2370592, upper bound: 111.1630521
time: 89.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 12, lower bound: -111.1983839, upper bound: 111.2017633
time: 95.03 seconds

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

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1649

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1554

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2494520, upper bound: 111.2569972
time: 95.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2485924, upper bound: 111.2578779
time: 156.60 seconds

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

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1764

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1735

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2311803, upper bound: 111.2527283
time: 94.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2445874, upper bound: 111.2389522
time: 91.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1546

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1705

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2476886, upper bound: 111.2392097
time: 90.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2394475, upper bound: 111.2474571
time: 92.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1303
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 905
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 912
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1397
type: RSZ, layer: 1, pos: 1497
type: RSZ, layer: 1, pos: 1004
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1389
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 962
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1010
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1496
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1334
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1323
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1437
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1430
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1317
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1461
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1302
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1421
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1353
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 973
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1352
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1318
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1479
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1588

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 927

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2578468, upper bound: 111.2361021
time: 139.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -111.2526861, upper bound: 111.2412590
time: 686.12 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 827.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2151369, upper bound: 111.2475110
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2223283, upper bound: 111.2403048
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2227553, upper bound: 111.2354828
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2113857, upper bound: 111.2468656
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2380581, upper bound: 111.2183929
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2323131, upper bound: 111.2241431
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2370592, upper bound: 111.1630521
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.1983839, upper bound: 111.2017633
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2494520, upper bound: 111.2569972
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2485924, upper bound: 111.2578779
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2311803, upper bound: 111.2527283
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2445874, upper bound: 111.2389522
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2476886, upper bound: 111.2392097
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2394475, upper bound: 111.2474571
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2578468, upper bound: 111.2361021
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 827.52
Output dim: 12, lower bound: -111.2526861, upper bound: 111.2412590
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=177.0996856689453
rel_dist={12: [-111.25869176025041, 111.25869175221396]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 12593.82 seconds
