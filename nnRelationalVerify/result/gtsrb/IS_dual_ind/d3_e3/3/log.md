## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 7200 seconds
Split limit: 100


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429)
1: (-57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768)
2: (-49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168)
3: (-55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182)
4: (-57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003)
5: (-58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783)
6: (-74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716)
7: (-70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945)
8: (-68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231)
9: (-55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713)
10: (-80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380)
11: (-87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477)
12: (-79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363)
13: (-79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273)
14: (-123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204)
15: (-65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714)
16: (-92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963)
17: (-127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067)
18: (-81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784)
19: (-63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985)
20: (-55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037)
21: (-77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207)
22: (-78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069)
23: (-64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089)
24: (-75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660)
25: (-64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663)
26: (-90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663)
27: (-78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616)
28: (-61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622)
29: (-84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274)
30: (-76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807)
31: (-81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393)
32: (-70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016)
33: (-101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296)
34: (-87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157)
35: (-84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095)
36: (-79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740)
37: (-117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118)
38: (-103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456)
39: (-118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528)
40: (-101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327)
41: (-73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230)
42: (-56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.79 + 119.39 = 122.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -76.2665171, upper bound: 76.2665171

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1460

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1753

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2646637, upper bound: 76.2199874
time: 1022.08 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2646637, upper bound: 76.2646636
time: 124.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1146.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1146.55
Output dim: 4, lower bound: -76.2646637, upper bound: 76.2199874
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1146.55
Output dim: 4, lower bound: -76.2646637, upper bound: 76.2646636

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -96.0363388, 66.8599854, -96.2191772, 66.9542313, -162.9905701, 163.0791626
1: -57.5919075, 59.3611488, -57.7378273, 59.4993744, -117.0912781, 117.0989761
2: -49.7093582, 51.9433556, -49.8490829, 52.0789185, -101.7882767, 101.7924347
3: -55.5595284, 62.6180229, -55.7104340, 62.7723732, -118.3318939, 118.3284454
4: -57.0523491, 59.9488640, -57.2446823, 60.1239510, -117.1763000, 117.1935425
5: -57.8860435, 63.5838814, -58.0179214, 63.7003326, -121.5863495, 121.6018066
6: -74.5378189, 51.6145134, -74.7273712, 51.7695847, -126.3073959, 126.3418884
7: -70.3967743, 65.8191528, -70.5539398, 65.9458466, -136.3426208, 136.3730927
8: -68.4279022, 71.4073944, -68.6631775, 71.6537781, -140.0816650, 140.0705719
9: -55.7660370, 60.1346397, -55.8443871, 60.2087746, -115.9748077, 115.9790268
10: -80.5112534, 75.7476273, -80.6165466, 75.8894806, -156.4007263, 156.3641510
11: -87.0625458, 61.9825668, -87.2546692, 62.1054382, -149.1679840, 149.2372437
12: -79.2822113, 69.6775970, -79.4137421, 69.8488464, -149.1310425, 149.0913391
13: -79.0794220, 89.0151672, -79.2213364, 89.2262344, -168.3056488, 168.2364960
14: -122.9185944, 60.3909760, -123.0870667, 60.5121078, -183.4306946, 183.4780426
15: -65.7288971, 58.5659981, -65.8584442, 58.6901779, -124.4190674, 124.4244308
16: -91.8958130, 67.9461746, -92.0607224, 68.0662079, -159.9620209, 160.0068970
17: -127.4767303, 86.2163239, -127.6174469, 86.3263016, -213.8030243, 213.8337708
18: -80.9692459, 65.2471008, -81.1799316, 65.4976730, -146.4669189, 146.4270325
19: -63.4296646, 36.5000801, -63.5561905, 36.5979195, -100.0275879, 100.0562592
20: -55.5446053, 43.9601402, -55.6253204, 44.0532227, -99.5978241, 99.5854568
21: -77.4138794, 44.9944763, -77.5485153, 45.0844917, -122.4983673, 122.5429916
22: -78.4440308, 49.6120682, -78.5645370, 49.6791878, -128.1231995, 128.1766052
23: -63.8415718, 44.6583862, -63.9753571, 44.7892570, -108.6308289, 108.6337433
24: -75.7325897, 42.3389053, -75.8845062, 42.3984680, -118.1310577, 118.2234039
25: -63.9761620, 50.0921288, -64.0850067, 50.1618805, -114.1380386, 114.1771393
26: -90.5684433, 73.4777374, -90.7323914, 73.6456375, -164.2140808, 164.2101288
27: -78.5254669, 49.7099609, -78.6580200, 49.7814445, -128.3069153, 128.3679810
28: -61.6616020, 51.4519157, -61.7797318, 51.5779228, -113.2395096, 113.2316437
29: -84.4094925, 49.2284813, -84.5533218, 49.2992668, -133.7087555, 133.7817993
30: -76.5152283, 54.8564682, -76.6551208, 54.9982262, -131.5134430, 131.5115662
31: -81.4138489, 45.6428299, -81.6086807, 45.7662582, -127.1800995, 127.2515030
32: -70.0621796, 53.2336311, -70.2016602, 53.3503761, -123.4125519, 123.4352875
33: -101.7612381, 75.9360352, -101.8679047, 76.0519257, -177.8131561, 177.8039398
34: -87.4777222, 58.8477402, -87.6227264, 58.9991455, -146.4768524, 146.4704590
35: -84.6898956, 59.4759979, -84.8094940, 59.5776787, -144.2675781, 144.2854767
36: -78.9558411, 61.2236328, -79.0583038, 61.3168945, -140.2727356, 140.2819366
37: -117.1671143, 64.9702530, -117.4263458, 65.1851044, -182.3522034, 182.3965912
38: -103.3576279, 77.2311401, -103.4743805, 77.3333740, -180.6909790, 180.7055206
39: -118.7200470, 75.5777740, -118.8459549, 75.6313477, -194.3513947, 194.4237366
40: -101.4099655, 61.7881088, -101.6405945, 62.0067177, -163.4166718, 163.4287109
41: -73.0544891, 51.1576729, -73.2406464, 51.2887154, -124.3432007, 124.3983154
42: -56.0798187, 48.1910515, -56.1947784, 48.3178902, -104.3977051, 104.3858337

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1460

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2517139, upper bound: 76.1869157
time: 97.96 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2517139, upper bound: 76.2070318
time: 122.77 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -96.2834167, 67.0206604, -96.2951660, 67.0287018, -163.3121185, 163.3158264
1: -57.7560997, 59.6072807, -57.7595062, 59.6165733, -117.3726501, 117.3667755
2: -49.8649635, 52.1843071, -49.8679047, 52.1937637, -102.0587158, 102.0522079
3: -55.7243423, 62.8901215, -55.7271080, 62.9007225, -118.6250610, 118.6172333
4: -57.2601242, 60.2623787, -57.2638435, 60.2738647, -117.5339890, 117.5262070
5: -58.0322227, 63.7858047, -58.0358429, 63.7945480, -121.8267670, 121.8216476
6: -74.8723831, 51.7957306, -74.8863144, 51.8011665, -126.6735535, 126.6820450
7: -70.5723572, 66.0386658, -70.5768509, 66.0492783, -136.6216431, 136.6155090
8: -68.6793671, 71.8497314, -68.6826019, 71.8655701, -140.5449371, 140.5323181
9: -55.8680305, 60.2487183, -55.8722115, 60.2548027, -116.1228256, 116.1209259
10: -80.6607056, 75.9228363, -80.6665268, 75.9277954, -156.5885010, 156.5893555
11: -87.3875732, 62.1330338, -87.4007034, 62.1388893, -149.5264587, 149.5337372
12: -79.5110245, 69.8759613, -79.5214844, 69.8810043, -149.3920135, 149.3974304
13: -79.2528534, 89.3761292, -79.2587738, 89.3890305, -168.6418762, 168.6349030
14: -123.1440353, 60.5891457, -123.1519547, 60.5986176, -183.7426453, 183.7410889
15: -65.8975601, 58.7543449, -65.9048920, 58.7700005, -124.6675568, 124.6592407
16: -92.1363525, 68.0937195, -92.1575623, 68.0983276, -160.2346802, 160.2512817
17: -127.6841888, 86.3753891, -127.6937027, 86.3817596, -214.0659180, 214.0690765
18: -81.3374481, 65.5187912, -81.3501740, 65.5223083, -146.8597565, 146.8689575
19: -63.6425896, 36.6100883, -63.6507607, 36.6125145, -100.2550964, 100.2608490
20: -55.6710854, 44.0821609, -55.6775513, 44.0862656, -99.7573547, 99.7597122
21: -77.6320724, 45.1020012, -77.6404343, 45.1052704, -122.7373428, 122.7424316
22: -78.6208191, 49.7001686, -78.6300278, 49.7043419, -128.3251648, 128.3302002
23: -64.0688629, 44.8120842, -64.0772629, 44.8158379, -108.8846970, 108.8893433
24: -75.9775772, 42.4134979, -75.9863510, 42.4167595, -118.3943329, 118.3998413
25: -64.1408691, 50.1846542, -64.1504593, 50.1890945, -114.3299637, 114.3351059
26: -90.8461151, 73.6709595, -90.8580933, 73.6766815, -164.5227966, 164.5290527
27: -78.7272491, 49.8026123, -78.7360229, 49.8058205, -128.5330658, 128.5386353
28: -61.8628082, 51.6044273, -61.8708038, 51.6078911, -113.4706955, 113.4752197
29: -84.6167297, 49.3149567, -84.6258926, 49.3180313, -133.9347534, 133.9408569
30: -76.7346497, 55.0329590, -76.7426453, 55.0382004, -131.7728577, 131.7756042
31: -81.7444611, 45.7823410, -81.7568359, 45.7857780, -127.5302277, 127.5391693
32: -70.2911835, 53.3805466, -70.3006287, 53.3848495, -123.6760330, 123.6811752
33: -101.9238434, 76.0736313, -101.9335098, 76.0781250, -178.0019684, 178.0071411
34: -87.7218018, 59.0170670, -87.7300034, 59.0209465, -146.7427521, 146.7470703
35: -84.8886642, 59.5932693, -84.8963318, 59.5963440, -144.4850006, 144.4895935
36: -79.1238098, 61.3277817, -79.1311188, 61.3296661, -140.4534760, 140.4588928
37: -117.6154480, 65.1990967, -117.6321106, 65.2017212, -182.8171692, 182.8312073
38: -103.5361099, 77.3579865, -103.5451355, 77.3618622, -180.8979645, 180.9031219
39: -118.9022522, 75.6570129, -118.9101715, 75.6627502, -194.5649719, 194.5671844
40: -101.8046646, 62.0270767, -101.8194580, 62.0325012, -163.8371582, 163.8465271
41: -73.3839722, 51.3116837, -73.3970108, 51.3165131, -124.7004852, 124.7086945
42: -56.2803116, 48.3464317, -56.2893333, 48.3518639, -104.6321716, 104.6357574

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1460

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2517139, upper bound: 76.2316058
time: 122.88 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2517139, upper bound: 76.2517137
time: 1456.94 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 1582.14 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1582.14
Output dim: 4, lower bound: -76.2517139, upper bound: 76.1869157
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1582.14
Output dim: 4, lower bound: -76.2517139, upper bound: 76.2070318
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1582.14
Output dim: 4, lower bound: -76.2517139, upper bound: 76.2316058
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1582.14
Output dim: 4, lower bound: -76.2517139, upper bound: 76.2517137

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -95.9756699, 66.8459320, -96.0574570, 66.9167328, -162.8923950, 162.9033813
1: -57.5484009, 59.3498764, -57.6221924, 59.4692764, -117.0176773, 116.9720535
2: -49.6391449, 51.9323807, -49.6620903, 52.0495872, -101.6887283, 101.5944595
3: -55.4820061, 62.6013374, -55.5036545, 62.7278137, -118.2098236, 118.1049881
4: -56.9771576, 59.9352074, -57.0445518, 60.0874977, -117.0646515, 116.9797592
5: -57.8087158, 63.5679092, -57.8123970, 63.6575775, -121.4662933, 121.3802948
6: -74.5127258, 51.5826416, -74.6602478, 51.6845589, -126.1972809, 126.2428894
7: -70.3317566, 65.8062973, -70.3806610, 65.9115295, -136.2432861, 136.1869507
8: -68.3492203, 71.3912735, -68.4536438, 71.6107178, -139.9599304, 139.8449097
9: -55.7474365, 60.0822372, -55.7946968, 60.0698395, -115.8172760, 115.8769226
10: -80.4806213, 75.6093903, -80.5348434, 75.5209045, -156.0015259, 156.1442261
11: -87.0405884, 61.8794861, -87.1959763, 61.8303909, -148.8709717, 149.0754700
12: -79.2640839, 69.5294800, -79.3653259, 69.4532776, -148.7173615, 148.8947906
13: -79.0448761, 88.9762726, -79.1301727, 89.1219101, -168.1667786, 168.1064453
14: -122.8784485, 60.2939644, -122.9796982, 60.2526321, -183.1310730, 183.2736359
15: -65.6754761, 58.5431938, -65.7169418, 58.6290321, -124.3045044, 124.2601242
16: -91.8606567, 67.8762054, -91.9666595, 67.8800812, -159.7407379, 159.8428650
17: -127.4526978, 86.0968781, -127.5534821, 86.0071640, -213.4598694, 213.6503601
18: -80.9428864, 65.1868286, -81.1095886, 65.3364258, -146.2793121, 146.2963867
19: -63.4083252, 36.4640427, -63.4991646, 36.5021744, -99.9104996, 99.9632034
20: -55.5237236, 43.9268723, -55.5695534, 43.9644852, -99.4882050, 99.4964142
21: -77.3913040, 44.9323425, -77.4881592, 44.9189301, -122.3102341, 122.4205017
22: -78.4173584, 49.5551186, -78.4933929, 49.5284729, -127.9458160, 128.0485077
23: -63.8229561, 44.6280174, -63.9255943, 44.7085304, -108.5314865, 108.5536118
24: -75.7071838, 42.3267212, -75.8168488, 42.3658257, -118.0729980, 118.1435547
25: -63.9572296, 50.0583115, -64.0344925, 50.0717201, -114.0289459, 114.0928040
26: -90.5433807, 73.3718491, -90.6653137, 73.3627548, -163.9061279, 164.0371552
27: -78.4843292, 49.6964417, -78.5487289, 49.7453918, -128.2297211, 128.2451630
28: -61.6404648, 51.4354820, -61.7232056, 51.5343552, -113.1748123, 113.1586761
29: -84.3896942, 49.1561813, -84.5005112, 49.1065598, -133.4962463, 133.6566925
30: -76.4961090, 54.8069611, -76.6039886, 54.8663826, -131.3624878, 131.4109497
31: -81.3838654, 45.6071968, -81.5284424, 45.6712494, -127.0551147, 127.1356354
32: -70.0419922, 53.1903915, -70.1477509, 53.2356796, -123.2776642, 123.3381424
33: -101.6908035, 75.9151459, -101.6797104, 75.9961395, -177.6869507, 177.5948486
34: -87.4208221, 58.8268585, -87.4706116, 58.9435272, -146.3643494, 146.2974701
35: -84.6261597, 59.4607239, -84.6391754, 59.5367699, -144.1629333, 144.0998840
36: -78.9131317, 61.2056732, -78.9442291, 61.2690125, -140.1821442, 140.1499023
37: -117.1316681, 64.9436951, -117.3318710, 65.1147919, -182.2464447, 182.2755585
38: -103.2977676, 77.2124481, -103.3152237, 77.2835083, -180.5812683, 180.5276794
39: -118.6713867, 75.5624313, -118.7168350, 75.5903778, -194.2617645, 194.2792664
40: -101.3651657, 61.7738609, -101.5217056, 61.9693413, -163.3345032, 163.2955627
41: -73.0278931, 51.1337395, -73.1695099, 51.2254791, -124.2533722, 124.3032455
42: -56.0606270, 48.1286850, -56.1434555, 48.1553993, -104.2160263, 104.2721405

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1460

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2038354, upper bound: 76.1855719
time: 147.46 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2038354, upper bound: 76.1855719
time: 128.44 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -96.0157089, 66.8548889, -96.2534866, 67.0391312, -163.0548248, 163.1083679
1: -57.5769005, 59.3553429, -57.7484589, 59.5332527, -117.1101532, 117.1038055
2: -49.6902657, 51.9383583, -49.8500824, 52.1965294, -101.8867950, 101.7884369
3: -55.5374680, 62.6107025, -55.7107430, 62.9186592, -118.4561310, 118.3214417
4: -57.0321884, 59.9426270, -57.2514000, 60.2112885, -117.2434692, 117.1940308
5: -57.8688545, 63.5768623, -58.0290985, 63.8421707, -121.7110291, 121.6059570
6: -74.5283051, 51.5822525, -74.7807693, 51.7577515, -126.2860413, 126.3630142
7: -70.3781281, 65.8131256, -70.5867462, 65.9789124, -136.3570251, 136.3998718
8: -68.4059525, 71.4010010, -68.6645813, 71.7667007, -140.1726532, 140.0655823
9: -55.7586517, 60.1201019, -55.8895760, 60.2320137, -115.9906616, 116.0096664
10: -80.5007019, 75.7112427, -80.8662415, 75.8720093, -156.3727112, 156.5774841
11: -87.0524750, 61.9551849, -87.4878235, 62.0852470, -149.1377106, 149.4429932
12: -79.2735062, 69.6412964, -79.7146988, 69.8314056, -149.1049042, 149.3559875
13: -79.0512085, 89.0021820, -79.2103119, 89.2882309, -168.3394470, 168.2124939
14: -122.9027863, 60.3689156, -123.2687531, 60.4997177, -183.4024963, 183.6376648
15: -65.6878204, 58.5575218, -65.8489227, 58.7570190, -124.4448395, 124.4064484
16: -91.8807831, 67.9082413, -92.1469727, 68.0570831, -159.9378662, 160.0552063
17: -127.4659119, 86.1868973, -127.8030090, 86.3184280, -213.7843323, 213.9898987
18: -80.9593658, 65.2330933, -81.3251648, 65.5052948, -146.4646606, 146.5582581
19: -63.4224281, 36.4886093, -63.6512451, 36.5998878, -100.0223083, 100.1398544
20: -55.5377502, 43.9493065, -55.7325211, 44.0516510, -99.5893936, 99.6818161
21: -77.4051056, 44.9780350, -77.7267914, 45.0792007, -122.4842987, 122.7048187
22: -78.4231110, 49.5953941, -78.5921783, 49.6943817, -128.1174927, 128.1875610
23: -63.8350754, 44.6446304, -64.0409241, 44.7924614, -108.6275253, 108.6855545
24: -75.7203674, 42.3276749, -75.9232635, 42.3939018, -118.1142578, 118.2509232
25: -63.9645004, 50.0820656, -64.1165695, 50.1761093, -114.1406021, 114.1986237
26: -90.5573044, 73.4505768, -90.9638062, 73.6490402, -164.2063446, 164.4143829
27: -78.5112915, 49.6940727, -78.6881561, 49.7742386, -128.2855225, 128.3822327
28: -61.6527824, 51.4422607, -61.8142319, 51.5928688, -113.2456512, 113.2564926
29: -84.3997726, 49.2093849, -84.5911331, 49.2927132, -133.6924896, 133.8005066
30: -76.5060272, 54.8310089, -76.7104568, 54.9902077, -131.4962311, 131.5414581
31: -81.4046860, 45.6311684, -81.7320480, 45.7666702, -127.1713562, 127.3632050
32: -70.0534668, 53.2187881, -70.2479095, 53.3565674, -123.4100342, 123.4666977
33: -101.7417984, 75.9279480, -101.8707275, 76.1894455, -177.9312286, 177.7986755
34: -87.4607773, 58.8400879, -87.6247025, 59.1019058, -146.5626831, 146.4647827
35: -84.6710358, 59.4708977, -84.8005676, 59.7429962, -144.4140167, 144.2714691
36: -78.9410095, 61.2180862, -79.0589142, 61.3781548, -140.3191681, 140.2770081
37: -117.1511841, 64.9507599, -117.4557037, 65.1974182, -182.3486023, 182.4064636
38: -103.3379440, 77.2236786, -103.4976196, 77.3789062, -180.7168579, 180.7212830
39: -118.7003403, 75.5718765, -118.8683395, 75.7038879, -194.4042358, 194.4402161
40: -101.3938217, 61.7619820, -101.6805267, 62.0025482, -163.3963623, 163.4425049
41: -73.0449371, 51.1330681, -73.2647247, 51.2935486, -124.3384781, 124.3977966
42: -56.0725784, 48.1634293, -56.2880478, 48.3176994, -104.3902740, 104.4514771

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1460

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2057371
time: 131.59 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2060429
time: 92.22 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -96.2227859, 67.0066528, -96.1334457, 66.9912033, -163.2139893, 163.1401062
1: -57.7126274, 59.5960121, -57.6438484, 59.5864906, -117.2991180, 117.2398605
2: -49.7947350, 52.1733437, -49.6809082, 52.1644440, -101.9591675, 101.8542480
3: -55.6468124, 62.8734703, -55.5203171, 62.8561707, -118.5029831, 118.3937759
4: -57.1849327, 60.2487602, -57.0637169, 60.2374077, -117.4223328, 117.3124771
5: -57.9548721, 63.7698555, -57.8303337, 63.7518005, -121.7066498, 121.6001892
6: -74.8473129, 51.7638245, -74.8191986, 51.7161598, -126.5634766, 126.5830078
7: -70.5073090, 66.0258560, -70.4035721, 66.0149536, -136.5222626, 136.4294128
8: -68.6007004, 71.8336334, -68.4730682, 71.8225250, -140.4232178, 140.3067017
9: -55.8493652, 60.1963387, -55.8224449, 60.1159058, -115.9652710, 116.0187836
10: -80.6300507, 75.7845993, -80.5846710, 75.5591736, -156.1892242, 156.3692627
11: -87.3656464, 62.0299034, -87.3420410, 61.8638725, -149.2294922, 149.3719482
12: -79.4929047, 69.7278442, -79.4730377, 69.4854584, -148.9783630, 149.2008820
13: -79.2182693, 89.3372879, -79.1675491, 89.2847290, -168.5029907, 168.5048370
14: -123.1038208, 60.4921150, -123.0444870, 60.3392334, -183.4430542, 183.5366058
15: -65.8441010, 58.7315331, -65.7633667, 58.7088203, -124.5529175, 124.4948959
16: -92.1012039, 68.0236816, -92.0634995, 67.9122009, -160.0133972, 160.0871582
17: -127.6601868, 86.2559738, -127.6297379, 86.0627899, -213.7229614, 213.8857117
18: -81.3111191, 65.4584808, -81.2798920, 65.3610687, -146.6721649, 146.7383728
19: -63.6212349, 36.5740128, -63.5937424, 36.5167885, -100.1380157, 100.1677551
20: -55.6502075, 44.0488853, -55.6218033, 43.9975243, -99.6477203, 99.6706848
21: -77.6094818, 45.0398788, -77.5800781, 44.9397163, -122.5491943, 122.6199570
22: -78.5941391, 49.6431885, -78.5588837, 49.5536156, -128.1477509, 128.2020569
23: -64.0502625, 44.7816696, -64.0274887, 44.7351036, -108.7853699, 108.8091583
24: -75.9521866, 42.4012833, -75.9187393, 42.3841324, -118.3363190, 118.3200226
25: -64.1219406, 50.1508102, -64.0998840, 50.0989799, -114.2209167, 114.2506943
26: -90.8210220, 73.5650482, -90.7910461, 73.3938446, -164.2148743, 164.3560791
27: -78.6861496, 49.7890854, -78.6267548, 49.7697372, -128.4558868, 128.4158325
28: -61.8416786, 51.5879745, -61.8143120, 51.5642853, -113.4059601, 113.4022827
29: -84.5969696, 49.2426605, -84.5730743, 49.1253090, -133.7222748, 133.8157349
30: -76.7155762, 54.9834404, -76.6915207, 54.9063339, -131.6219025, 131.6749573
31: -81.7144775, 45.7466507, -81.6766052, 45.6907463, -127.4052277, 127.4232483
32: -70.2710266, 53.3373413, -70.2467194, 53.2701683, -123.5411987, 123.5840607
33: -101.8534546, 76.0527039, -101.7453613, 76.0222473, -177.8757019, 177.7980652
34: -87.6649017, 58.9961433, -87.5779800, 58.9652863, -146.6301880, 146.5741272
35: -84.8249435, 59.5779572, -84.7260056, 59.5554008, -144.3803406, 144.3039551
36: -79.0811157, 61.3098068, -79.0170593, 61.2817726, -140.3628845, 140.3268738
37: -117.5800705, 65.1724930, -117.5376816, 65.1313934, -182.7114563, 182.7101746
38: -103.4763489, 77.3392334, -103.3860703, 77.3119354, -180.7882843, 180.7252808
39: -118.8535309, 75.6416855, -118.7810516, 75.6217804, -194.4753113, 194.4227295
40: -101.7599335, 62.0128021, -101.7007446, 61.9951096, -163.7550354, 163.7135468
41: -73.3573532, 51.2877197, -73.3259201, 51.2532730, -124.6106262, 124.6136398
42: -56.2611275, 48.2840271, -56.2379913, 48.1893959, -104.4505234, 104.5220184

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1460

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2302568
time: 159.14 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2306118
time: 127.55 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -96.2627869, 67.0155792, -96.3293686, 67.1136627, -163.3764496, 163.3449402
1: -57.7410851, 59.6014862, -57.7701263, 59.6504669, -117.3915558, 117.3716125
2: -49.8458481, 52.1792831, -49.8689079, 52.3114319, -102.1572800, 102.0481796
3: -55.7022438, 62.8828049, -55.7274094, 63.0470390, -118.7492599, 118.6102142
4: -57.2399483, 60.2561493, -57.2705536, 60.3612061, -117.6011505, 117.5266953
5: -58.0149956, 63.7787781, -58.0470085, 63.9363937, -121.9513855, 121.8257751
6: -74.8628845, 51.7634659, -74.9397430, 51.7893410, -126.6522217, 126.7031860
7: -70.5536575, 66.0326538, -70.6096344, 66.0823517, -136.6360168, 136.6422882
8: -68.6574249, 71.8433533, -68.6840057, 71.9785995, -140.6360168, 140.5273590
9: -55.8606110, 60.2341766, -55.9173889, 60.2780571, -116.1386719, 116.1515656
10: -80.6501617, 75.8864441, -80.9162292, 75.9103012, -156.5604553, 156.8026733
11: -87.3775101, 62.1056175, -87.6339264, 62.1186447, -149.4961548, 149.7395477
12: -79.5023193, 69.8396683, -79.8224640, 69.8635712, -149.3658752, 149.6621094
13: -79.2246094, 89.3631287, -79.2476654, 89.4510193, -168.6756134, 168.6107941
14: -123.1281891, 60.5670547, -123.3335876, 60.5862198, -183.7144165, 183.9006348
15: -65.8564911, 58.7458611, -65.8953247, 58.8368378, -124.6933289, 124.6411819
16: -92.1213531, 68.0557556, -92.2438507, 68.0891724, -160.2105255, 160.2996063
17: -127.6734009, 86.3459320, -127.8793106, 86.3739471, -214.0473480, 214.2252350
18: -81.3276367, 65.5047760, -81.4954681, 65.5299149, -146.8575287, 147.0002441
19: -63.6353302, 36.5985641, -63.7458153, 36.6145020, -100.2498169, 100.3443756
20: -55.6642342, 44.0713120, -55.7848015, 44.0846863, -99.7489166, 99.8561096
21: -77.6233063, 45.0855827, -77.8187408, 45.0999908, -122.7232971, 122.9043274
22: -78.5998993, 49.6834869, -78.6576691, 49.7195168, -128.3193970, 128.3411560
23: -64.0623550, 44.7982903, -64.1427917, 44.8190079, -108.8813629, 108.9410782
24: -75.9653625, 42.4022636, -76.0251389, 42.4122238, -118.3775864, 118.4273911
25: -64.1292419, 50.1745720, -64.1819305, 50.2033195, -114.3325577, 114.3565063
26: -90.8349609, 73.6437531, -91.0895538, 73.6800537, -164.5149994, 164.7333069
27: -78.7130737, 49.7867432, -78.7661438, 49.7985573, -128.5116272, 128.5528870
28: -61.8539810, 51.5947723, -61.9053116, 51.6228447, -113.4768219, 113.5000839
29: -84.6070175, 49.2958717, -84.6637344, 49.3114662, -133.9184875, 133.9595947
30: -76.7254791, 55.0075073, -76.7979736, 55.0301056, -131.7555847, 131.8054810
31: -81.7353058, 45.7706261, -81.8802414, 45.7861862, -127.5214920, 127.6508636
32: -70.2824860, 53.3657227, -70.3468933, 53.3910980, -123.6735840, 123.7126160
33: -101.9044113, 76.0655594, -101.9363556, 76.2155914, -178.1199951, 178.0019226
34: -87.7048187, 59.0094528, -87.7320480, 59.1237106, -146.8285217, 146.7415009
35: -84.8698196, 59.5881577, -84.8873520, 59.7616425, -144.6314697, 144.4755096
36: -79.1090317, 61.3222198, -79.1317444, 61.3909225, -140.4999542, 140.4539642
37: -117.5995255, 65.1795807, -117.6614380, 65.2140198, -182.8135376, 182.8410187
38: -103.5164795, 77.3504944, -103.5683136, 77.4073792, -180.9238586, 180.9188080
39: -118.8825073, 75.6511459, -118.9325104, 75.7352600, -194.6177673, 194.5836487
40: -101.7885590, 62.0009537, -101.8594055, 62.0283432, -163.8168945, 163.8603516
41: -73.3744049, 51.2870445, -73.4210815, 51.3213730, -124.6957779, 124.7081299
42: -56.2730789, 48.3187981, -56.3825912, 48.3516617, -104.6247177, 104.7013855

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1460

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2504326
time: 338.25 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2507255
time: 147.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 488.31 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 488.31
Output dim: 4, lower bound: -76.2038354, upper bound: 76.1855719
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 488.31
Output dim: 4, lower bound: -76.2038354, upper bound: 76.1855719
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 488.31
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2057371
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 488.31
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2060429
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 488.31
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2302568
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 488.31
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2306118
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 488.31
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2504326
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 488.31
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2507255

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -95.7071228, 66.6936340, -95.9438858, 66.8979034, -162.6050262, 162.6375122
1: -57.3076019, 59.2042923, -57.5067825, 59.4511147, -116.7587128, 116.7110672
2: -49.4033279, 51.7905502, -49.5493317, 52.0315018, -101.4348297, 101.3398743
3: -55.1864853, 62.4041252, -55.3621216, 62.7038155, -117.8903046, 117.7662354
4: -56.7096519, 59.7410278, -56.9160347, 60.0582848, -116.7679367, 116.6570435
5: -57.5159454, 63.3628502, -57.6714706, 63.6334000, -121.1493454, 121.0343170
6: -74.3235245, 51.4782791, -74.5893707, 51.6538315, -125.9773560, 126.0676346
7: -70.0251465, 65.6294556, -70.2319260, 65.8927460, -135.9178925, 135.8613892
8: -68.0669479, 71.2111664, -68.3193817, 71.5862732, -139.6532135, 139.5305328
9: -55.5090714, 59.9387283, -55.6848106, 60.0469284, -115.5559998, 115.6235275
10: -80.2666397, 75.4865036, -80.4447784, 75.4780426, -155.7446899, 155.9312744
11: -86.8324356, 61.7297478, -87.1410675, 61.7619858, -148.5944214, 148.8708191
12: -79.0889053, 69.3062592, -79.3298874, 69.3510971, -148.4400024, 148.6361389
13: -78.8034821, 88.7884979, -79.0177383, 89.0850220, -167.8885040, 167.8062439
14: -122.6504211, 60.1541672, -122.8970566, 60.1904984, -182.8408813, 183.0512238
15: -65.4975128, 58.4168320, -65.6347427, 58.6006126, -124.0981140, 124.0515747
16: -91.5734787, 67.7533875, -91.8435516, 67.8596802, -159.4331512, 159.5969391
17: -127.2146149, 85.9322433, -127.4653015, 85.9386215, -213.1532288, 213.3975525
18: -80.7808685, 64.8960724, -81.0774612, 65.1994019, -145.9802704, 145.9735260
19: -63.2446404, 36.2619286, -63.4689026, 36.4031982, -99.6478424, 99.7308350
20: -55.4136467, 43.7777710, -55.5425987, 43.8964539, -99.3101044, 99.3203735
21: -77.2263489, 44.7570877, -77.4455414, 44.8357124, -122.0620575, 122.2026291
22: -78.2388840, 49.3447647, -78.4589386, 49.4298172, -127.6687012, 127.8036957
23: -63.6205750, 44.3583755, -63.8924408, 44.5782738, -108.1988449, 108.2508087
24: -75.4933853, 42.0927124, -75.7883377, 42.2527275, -117.7461090, 117.8810349
25: -63.7846909, 49.8124123, -64.0065002, 49.9535980, -113.7382889, 113.8189087
26: -90.3468094, 73.0420380, -90.6321335, 73.2077179, -163.5545349, 163.6741638
27: -78.3157654, 49.5188599, -78.5158539, 49.6624565, -127.9782257, 128.0347137
28: -61.4690514, 51.1967087, -61.6986504, 51.4203415, -112.8893814, 112.8953552
29: -84.1734009, 48.9849854, -84.4530334, 49.0247803, -133.1981659, 133.4380188
30: -76.3303375, 54.6245651, -76.5678101, 54.7820053, -131.1123352, 131.1923828
31: -81.1694641, 45.3802948, -81.4918900, 45.5614929, -126.7309570, 126.8721848
32: -69.8901825, 53.1030922, -70.1052399, 53.1971550, -123.0873413, 123.2083282
33: -101.5263901, 75.7375717, -101.6298752, 75.9160919, -177.4424744, 177.3674469
34: -87.2463226, 58.5663147, -87.4349060, 58.8179169, -146.0642395, 146.0012207
35: -84.4809952, 59.2545357, -84.6064758, 59.4402771, -143.9212646, 143.8610077
36: -78.7834930, 61.0069656, -78.9156494, 61.1743164, -139.9578094, 139.9226074
37: -116.8510590, 64.6674347, -117.2705612, 64.9837952, -181.8348541, 181.9379883
38: -103.1075897, 76.9709091, -103.2738647, 77.1734314, -180.2810211, 180.2447815
39: -118.4562912, 75.4157639, -118.6591568, 75.5225525, -193.9788513, 194.0749054
40: -101.1732254, 61.6979599, -101.4679108, 61.9358368, -163.1090698, 163.1658630
41: -72.8698120, 51.0277939, -73.1237564, 51.1792603, -124.0490723, 124.1515350
42: -55.9124794, 48.0530243, -56.0906944, 48.1247711, -104.0372467, 104.1437225

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1460

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.1837900
time: 113.81 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2022931, upper bound: 76.1840278
time: 94.88 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -95.9638062, 66.8379974, -96.0520325, 66.9132080, -162.8770142, 162.8900146
1: -57.5400429, 59.3434181, -57.6184196, 59.4664078, -117.0064468, 116.9618378
2: -49.6315346, 51.9266891, -49.6586761, 52.0470200, -101.6785583, 101.5853577
3: -55.4726639, 62.5941658, -55.4994965, 62.7246132, -118.1972809, 118.0936584
4: -56.9684563, 59.9257393, -57.0406723, 60.0832291, -117.0516815, 116.9664154
5: -57.7990761, 63.5612106, -57.8080978, 63.6545525, -121.4536285, 121.3693085
6: -74.4942322, 51.5782013, -74.6518402, 51.6824989, -126.1767273, 126.2300415
7: -70.3212280, 65.7998352, -70.3759460, 65.9086456, -136.2298737, 136.1757812
8: -68.3391800, 71.3840027, -68.4491119, 71.6074448, -139.9466095, 139.8330994
9: -55.7385254, 60.0735855, -55.7906952, 60.0659180, -115.8044281, 115.8642807
10: -80.4714890, 75.5831909, -80.5306854, 75.5092773, -155.9807739, 156.1138763
11: -87.0313110, 61.8637161, -87.1917725, 61.8230515, -148.8543701, 149.0554810
12: -79.2537308, 69.5213013, -79.3606339, 69.4495850, -148.7033081, 148.8819275
13: -79.0342255, 88.9672775, -79.1254425, 89.1178131, -168.1520386, 168.0927124
14: -122.8651505, 60.2861633, -122.9737167, 60.2490540, -183.1141968, 183.2598724
15: -65.6674805, 58.5355644, -65.7133865, 58.6256409, -124.2931213, 124.2489471
16: -91.8487167, 67.8681946, -91.9612045, 67.8764648, -159.7251740, 159.8294067
17: -127.4401169, 86.0779266, -127.5478134, 85.9985046, -213.4386292, 213.6257324
18: -80.9363556, 65.1761093, -81.1066589, 65.3316956, -146.2680359, 146.2827759
19: -63.4018555, 36.4559669, -63.4962807, 36.4985542, -99.9004059, 99.9522400
20: -55.5168419, 43.9208908, -55.5664902, 43.9617615, -99.4785919, 99.4873810
21: -77.3828201, 44.9240837, -77.4843597, 44.9151764, -122.2979965, 122.4084473
22: -78.4094467, 49.5479240, -78.4898834, 49.5252342, -127.9346771, 128.0378113
23: -63.8154945, 44.6187706, -63.9221954, 44.7044144, -108.5199127, 108.5409698
24: -75.6991959, 42.3187561, -75.8132477, 42.3622704, -118.0614624, 118.1320038
25: -63.9501419, 50.0492249, -64.0313263, 50.0676422, -114.0177841, 114.0805511
26: -90.5339661, 73.3609619, -90.6611023, 73.3579102, -163.8918762, 164.0220642
27: -78.4741364, 49.6902847, -78.5441513, 49.7426834, -128.2168274, 128.2344360
28: -61.6343422, 51.4274139, -61.7204933, 51.5307312, -113.1650543, 113.1479034
29: -84.3781433, 49.1490860, -84.4953766, 49.1028175, -133.4809570, 133.6444702
30: -76.4863434, 54.7998581, -76.5996094, 54.8631783, -131.3494873, 131.3994751
31: -81.3763962, 45.5975990, -81.5251236, 45.6669350, -127.0433273, 127.1227188
32: -70.0304031, 53.1830254, -70.1424942, 53.2323952, -123.2627869, 123.3255157
33: -101.6812820, 75.9069824, -101.6753998, 75.9924850, -177.6737671, 177.5823669
34: -87.4120636, 58.8172493, -87.4665527, 58.9392166, -146.3512878, 146.2837982
35: -84.6195526, 59.4524536, -84.6362152, 59.5330124, -144.1525574, 144.0886688
36: -78.9032669, 61.1986160, -78.9397888, 61.2658157, -140.1690674, 140.1383972
37: -117.1171265, 64.9350586, -117.3251953, 65.1109467, -182.2280731, 182.2602386
38: -103.2857361, 77.2029648, -103.3098145, 77.2792511, -180.5649719, 180.5127716
39: -118.6606369, 75.5552139, -118.7120438, 75.5871429, -194.2477722, 194.2672424
40: -101.3509216, 61.7677307, -101.5151596, 61.9665985, -163.3175201, 163.2828827
41: -73.0165710, 51.1153412, -73.1642456, 51.2162170, -124.2327881, 124.2795868
42: -56.0489731, 48.1163979, -56.1380424, 48.1500053, -104.1989746, 104.2544250

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1460

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.1841428
time: 117.08 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.1843825
time: 139.88 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -95.7471313, 66.7025452, -96.1398926, 67.0203247, -162.7674561, 162.8424377
1: -57.3361130, 59.2097816, -57.6330833, 59.5150795, -116.8511887, 116.8428650
2: -49.4544373, 51.7965240, -49.7373352, 52.1785240, -101.6329651, 101.5338593
3: -55.2419319, 62.4134865, -55.5692177, 62.8946953, -118.1366272, 117.9827042
4: -56.7646904, 59.7484474, -57.1229057, 60.1821022, -116.9467773, 116.8713531
5: -57.5760994, 63.3718109, -57.8881950, 63.8180008, -121.3940811, 121.2600098
6: -74.3390808, 51.4779091, -74.7099075, 51.7270432, -126.0661240, 126.1878204
7: -70.0715332, 65.6363068, -70.4380035, 65.9601669, -136.0317078, 136.0743103
8: -68.1236649, 71.2209015, -68.5303650, 71.7423248, -139.8659973, 139.7512665
9: -55.5202675, 59.9766045, -55.7797432, 60.2090836, -115.7293549, 115.7563477
10: -80.2866592, 75.5883331, -80.7762909, 75.8290787, -156.1157379, 156.3646240
11: -86.8443451, 61.8054428, -87.4329224, 62.0167770, -148.8611145, 149.2383728
12: -79.0983734, 69.4180756, -79.6792908, 69.7292404, -148.8276062, 149.0973663
13: -78.8097839, 88.8143921, -79.0978699, 89.2513199, -168.0610962, 167.9122620
14: -122.6747665, 60.2291031, -123.1862259, 60.4375954, -183.1123657, 183.4153290
15: -65.5098724, 58.4311943, -65.7667160, 58.7286949, -124.2385712, 124.1979065
16: -91.5936432, 67.7854156, -92.0239410, 68.0366440, -159.6302795, 159.8093567
17: -127.2278214, 86.0222321, -127.7149200, 86.2498474, -213.4776611, 213.7371521
18: -80.7974091, 64.9422913, -81.2930145, 65.3682480, -146.1656494, 146.2353058
19: -63.2587509, 36.2864609, -63.6210098, 36.5009270, -99.7596741, 99.9074707
20: -55.4276772, 43.8002052, -55.7056198, 43.9836044, -99.4112854, 99.5058289
21: -77.2401810, 44.8027916, -77.6841965, 44.9959869, -122.2361679, 122.4869843
22: -78.2446289, 49.3850555, -78.5576782, 49.5956879, -127.8403168, 127.9427338
23: -63.6326942, 44.3749771, -64.0077515, 44.6622047, -108.2948990, 108.3827286
24: -75.5065842, 42.0936852, -75.8947296, 42.2808113, -117.7873993, 117.9884033
25: -63.7919846, 49.8361778, -64.0885773, 50.0579720, -113.8499451, 113.9247513
26: -90.3607559, 73.1206894, -90.9306717, 73.4938812, -163.8546295, 164.0513611
27: -78.3426895, 49.5164757, -78.6552277, 49.6912804, -128.0339661, 128.1717072
28: -61.4813995, 51.2034988, -61.7896385, 51.4788589, -112.9602585, 112.9931335
29: -84.1835251, 49.0381851, -84.5436707, 49.2109489, -133.3944702, 133.5818481
30: -76.3402710, 54.6486282, -76.6742859, 54.9057999, -131.2460632, 131.3229065
31: -81.1903076, 45.4042511, -81.6955109, 45.6568985, -126.8472061, 127.0997620
32: -69.9016800, 53.1314812, -70.2054138, 53.3180504, -123.2197266, 123.3368988
33: -101.5773849, 75.7504120, -101.8208618, 76.1094208, -177.6867981, 177.5712585
34: -87.2862701, 58.5795441, -87.5889664, 58.9763718, -146.2626343, 146.1685028
35: -84.5258713, 59.2646866, -84.7678528, 59.6465073, -144.1723633, 144.0325317
36: -78.8113861, 61.0193596, -79.0303497, 61.2835007, -140.0948792, 140.0497131
37: -116.8705139, 64.6744690, -117.3943481, 65.0663834, -181.9368896, 182.0688019
38: -103.1477737, 76.9821625, -103.4561615, 77.2688293, -180.4165955, 180.4383240
39: -118.4852676, 75.4252625, -118.8106079, 75.6360397, -194.1212921, 194.2358704
40: -101.2018814, 61.6860771, -101.6266937, 61.9690399, -163.1709290, 163.3127747
41: -72.8868866, 51.0270996, -73.2189713, 51.2473297, -124.1342163, 124.2460709
42: -55.9244461, 48.0877380, -56.2353439, 48.2869949, -104.2114410, 104.3230743

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1460

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2039682
time: 98.82 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2022931, upper bound: 76.2041949
time: 145.89 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -96.0038300, 66.8468781, -96.2480850, 67.0355911, -163.0394287, 163.0949554
1: -57.5685158, 59.3489037, -57.7446823, 59.5303612, -117.0988617, 117.0935822
2: -49.6826515, 51.9326363, -49.8466721, 52.1939659, -101.8766174, 101.7793045
3: -55.5281258, 62.6035156, -55.7065849, 62.9154549, -118.4435730, 118.3101044
4: -57.0234718, 59.9331398, -57.2475357, 60.2069893, -117.2304535, 117.1806793
5: -57.8591995, 63.5701370, -58.0248184, 63.8391418, -121.6983414, 121.5949478
6: -74.5098114, 51.5778236, -74.7723999, 51.7556839, -126.2654953, 126.3502197
7: -70.3675995, 65.8066254, -70.5820160, 65.9760437, -136.3436432, 136.3886414
8: -68.3959198, 71.3937149, -68.6600876, 71.7634430, -140.1593628, 140.0538025
9: -55.7497826, 60.1114388, -55.8855743, 60.2280922, -115.9778595, 115.9970093
10: -80.4915543, 75.6850510, -80.8620758, 75.8603897, -156.3519440, 156.5471191
11: -87.0432053, 61.9394455, -87.4836273, 62.0779037, -149.1211090, 149.4230652
12: -79.2631226, 69.6331253, -79.7100220, 69.8277130, -149.0908356, 149.3431396
13: -79.0405502, 88.9931335, -79.2055817, 89.2841187, -168.3246613, 168.1987000
14: -122.8894196, 60.3610306, -123.2627869, 60.4961090, -183.3854980, 183.6237946
15: -65.6798553, 58.5499229, -65.8453445, 58.7536278, -124.4334869, 124.3952637
16: -91.8688507, 67.9002304, -92.1415405, 68.0534668, -159.9223175, 160.0417786
17: -127.4532852, 86.1679153, -127.7973328, 86.3097534, -213.7630310, 213.9652405
18: -80.9528656, 65.2224121, -81.3222046, 65.5005722, -146.4534302, 146.5446167
19: -63.4159431, 36.4805222, -63.6483650, 36.5962601, -100.0121994, 100.1288757
20: -55.5308647, 43.9433212, -55.7294617, 44.0489349, -99.5798035, 99.6727829
21: -77.3966217, 44.9697762, -77.7230072, 45.0754700, -122.4720917, 122.6927795
22: -78.4151917, 49.5882568, -78.5886536, 49.6911316, -128.1063232, 128.1769104
23: -63.8276024, 44.6353836, -64.0375061, 44.7883453, -108.6159515, 108.6728897
24: -75.7123718, 42.3197136, -75.9196625, 42.3903427, -118.1027145, 118.2393646
25: -63.9574051, 50.0730019, -64.1134033, 50.1720009, -114.1294022, 114.1864014
26: -90.5479279, 73.4396286, -90.9595795, 73.6441650, -164.1920776, 164.3992004
27: -78.5010834, 49.6879082, -78.6835785, 49.7715225, -128.2726135, 128.3714752
28: -61.6466675, 51.4341888, -61.8115311, 51.5892410, -113.2359085, 113.2457199
29: -84.3882446, 49.2023125, -84.5859985, 49.2890053, -133.6772461, 133.7883148
30: -76.4962616, 54.8239098, -76.7060928, 54.9869766, -131.4832458, 131.5299988
31: -81.3972321, 45.6215515, -81.7287140, 45.7623444, -127.1595764, 127.3502579
32: -70.0418777, 53.2114143, -70.2426453, 53.3532677, -123.3951416, 123.4540558
33: -101.7322617, 75.9198227, -101.8664169, 76.1857758, -177.9180298, 177.7862396
34: -87.4520416, 58.8304672, -87.6206741, 59.0976219, -146.5496521, 146.4511414
35: -84.6644440, 59.4625969, -84.7976151, 59.7392311, -144.4036713, 144.2602081
36: -78.9311447, 61.2109909, -79.0544739, 61.3749847, -140.3061218, 140.2654572
37: -117.1365967, 64.9421234, -117.4490433, 65.1935883, -182.3301849, 182.3911743
38: -103.3258972, 77.2141724, -103.4921570, 77.3746643, -180.7005615, 180.7063293
39: -118.6896286, 75.5646820, -118.8635025, 75.7006378, -194.3902588, 194.4281921
40: -101.3795700, 61.7558556, -101.6739426, 61.9998169, -163.3793945, 163.4297943
41: -73.0336304, 51.1146660, -73.2594757, 51.2842827, -124.3179016, 124.3741302
42: -56.0609207, 48.1511421, -56.2826424, 48.3123093, -104.3732300, 104.4337845

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1460

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2042811
time: 164.24 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2045007
time: 113.85 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -95.9542999, 66.8543167, -96.0198517, 66.9723969, -162.9266968, 162.8741760
1: -57.4718132, 59.4504433, -57.5284348, 59.5683289, -117.0401459, 116.9788818
2: -49.5589294, 52.0315018, -49.5681648, 52.1463661, -101.7052765, 101.5996704
3: -55.3512726, 62.6762505, -55.3787994, 62.8321953, -118.1834412, 118.0550537
4: -56.9174194, 60.0545807, -56.9351807, 60.2081909, -117.1256104, 116.9897614
5: -57.6621475, 63.5647964, -57.6894035, 63.7276192, -121.3897705, 121.2541962
6: -74.6581879, 51.6593170, -74.7483673, 51.6854019, -126.3435822, 126.4076767
7: -70.2007217, 65.8490143, -70.2548294, 65.9961548, -136.1968689, 136.1038513
8: -68.3184357, 71.6535339, -68.3388596, 71.7980728, -140.1165161, 139.9924011
9: -55.6110001, 60.0528030, -55.7125511, 60.0930023, -115.7039948, 115.7653427
10: -80.4160385, 75.6616058, -80.4946136, 75.5163116, -155.9323425, 156.1562195
11: -87.1575699, 61.8802147, -87.2871704, 61.7955132, -148.9530792, 149.1673889
12: -79.3177338, 69.5046387, -79.4376068, 69.3833694, -148.7011108, 148.9422302
13: -78.9768143, 89.1494751, -79.0551224, 89.2478027, -168.2246094, 168.2045898
14: -122.8755646, 60.3524551, -122.9617844, 60.2771416, -183.1527100, 183.3142395
15: -65.6661682, 58.6051598, -65.6811447, 58.6803894, -124.3465576, 124.2863007
16: -91.8141632, 67.9007111, -91.9404907, 67.8917694, -159.7059326, 159.8411865
17: -127.4220047, 86.0913391, -127.5415115, 85.9941711, -213.4161682, 213.6328278
18: -81.1491394, 65.1676865, -81.2477570, 65.2240219, -146.3731537, 146.4154358
19: -63.4575577, 36.3719101, -63.5634575, 36.4178391, -99.8753815, 99.9353638
20: -55.5401535, 43.8997993, -55.5948410, 43.9294891, -99.4696426, 99.4946442
21: -77.4445343, 44.8646240, -77.5374451, 44.8564949, -122.3010254, 122.4020691
22: -78.4155884, 49.4328461, -78.5243683, 49.4549713, -127.8705597, 127.9572144
23: -63.8478775, 44.5120010, -63.9943237, 44.6048279, -108.4526978, 108.5063248
24: -75.7384644, 42.1673126, -75.8902435, 42.2710495, -118.0095062, 118.0575562
25: -63.9493446, 49.9049072, -64.0718765, 49.9808044, -113.9301453, 113.9767761
26: -90.6244583, 73.2351837, -90.7578430, 73.2387695, -163.8632202, 163.9930115
27: -78.5175171, 49.6115112, -78.5938568, 49.6867752, -128.2042847, 128.2053680
28: -61.6702538, 51.3492126, -61.7897491, 51.4502640, -113.1205139, 113.1389618
29: -84.3806534, 49.0714760, -84.5256042, 49.0435371, -133.4241943, 133.5970764
30: -76.5498657, 54.8010254, -76.6554184, 54.8219604, -131.3718262, 131.4564362
31: -81.5000992, 45.5197678, -81.6400528, 45.5809860, -127.0810852, 127.1598206
32: -70.1192169, 53.2499275, -70.2041855, 53.2316208, -123.3508377, 123.4540939
33: -101.6889877, 75.8751450, -101.6955490, 75.9422226, -177.6311951, 177.5706940
34: -87.4904480, 58.7355728, -87.5422897, 58.8396645, -146.3301086, 146.2778625
35: -84.6797485, 59.3717384, -84.6932983, 59.4588852, -144.1386108, 144.0650330
36: -78.9514999, 61.1110497, -78.9884949, 61.1870422, -140.1385498, 140.0995483
37: -117.2994766, 64.8962555, -117.4764099, 65.0004120, -182.2998810, 182.3726654
38: -103.2861862, 77.0976868, -103.3447342, 77.2018127, -180.4879761, 180.4424133
39: -118.6382751, 75.4950409, -118.7233276, 75.5539703, -194.1922455, 194.2183533
40: -101.5679932, 61.9368858, -101.6469269, 61.9615936, -163.5295868, 163.5838013
41: -73.1992950, 51.1817322, -73.2801590, 51.2070885, -124.4063721, 124.4618835
42: -56.1130066, 48.2083168, -56.1852455, 48.1587448, -104.2717514, 104.3935623

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1460

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2285201
time: 129.97 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2022931, upper bound: 76.2287142
time: 124.63 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -96.2109299, 66.9986954, -96.1280212, 66.9876862, -163.1986084, 163.1267090
1: -57.7042351, 59.5895882, -57.6400795, 59.5836143, -117.2878342, 117.2296600
2: -49.7871323, 52.1676331, -49.6775017, 52.1618805, -101.9490128, 101.8451233
3: -55.6374397, 62.8662834, -55.5161629, 62.8529854, -118.4904175, 118.3824463
4: -57.1762428, 60.2392921, -57.0598335, 60.2331238, -117.4093552, 117.2991180
5: -57.9452667, 63.7631416, -57.8260307, 63.7487869, -121.6940460, 121.5891724
6: -74.8288116, 51.7593803, -74.8108063, 51.7141037, -126.5429077, 126.5701904
7: -70.4968033, 66.0193863, -70.3988647, 66.0120773, -136.5088806, 136.4182434
8: -68.5906525, 71.8263550, -68.4685211, 71.8192444, -140.4098816, 140.2948761
9: -55.8404694, 60.1876717, -55.8184395, 60.1119881, -115.9524536, 116.0061035
10: -80.6209106, 75.7583847, -80.5805283, 75.5475616, -156.1684723, 156.3389130
11: -87.3563690, 62.0141373, -87.3378143, 61.8564987, -149.2128448, 149.3519592
12: -79.4825363, 69.7196350, -79.4683762, 69.4817886, -148.9643097, 149.1880188
13: -79.2076263, 89.3282471, -79.1628342, 89.2806244, -168.4882355, 168.4910889
14: -123.0904999, 60.4842682, -123.0385132, 60.3355942, -183.4260864, 183.5227814
15: -65.8361511, 58.7239304, -65.7597885, 58.7053909, -124.5415421, 124.4837189
16: -92.0892563, 68.0156479, -92.0580673, 67.9085693, -159.9978180, 160.0737152
17: -127.6475449, 86.2369919, -127.6240082, 86.0541000, -213.7016449, 213.8609924
18: -81.3045807, 65.4478149, -81.2769318, 65.3563080, -146.6608887, 146.7247467
19: -63.6147728, 36.5659561, -63.5908508, 36.5131683, -100.1279221, 100.1568069
20: -55.6433334, 44.0429153, -55.6187477, 43.9947968, -99.6381302, 99.6616669
21: -77.6010208, 45.0316391, -77.5762939, 44.9359779, -122.5370026, 122.6079330
22: -78.5862350, 49.6360321, -78.5553436, 49.5503883, -128.1366119, 128.1913757
23: -64.0428009, 44.7724152, -64.0241089, 44.7309952, -108.7737885, 108.7965240
24: -75.9442062, 42.3933105, -75.9151306, 42.3805847, -118.3247833, 118.3084412
25: -64.1148529, 50.1417160, -64.0967331, 50.0948601, -114.2097092, 114.2384415
26: -90.8116226, 73.5541382, -90.7868195, 73.3889923, -164.2006073, 164.3409576
27: -78.6759338, 49.7829285, -78.6221695, 49.7670135, -128.4429321, 128.4050903
28: -61.8355408, 51.5799065, -61.8115807, 51.5606651, -113.3962097, 113.3914871
29: -84.5853882, 49.2355537, -84.5679321, 49.1215782, -133.7069702, 133.8034821
30: -76.7057953, 54.9763451, -76.6871567, 54.9031334, -131.6089172, 131.6634979
31: -81.7070312, 45.7370605, -81.6732635, 45.6864319, -127.3934555, 127.4103241
32: -70.2594376, 53.3299446, -70.2414551, 53.2668610, -123.5262985, 123.5713806
33: -101.8439178, 76.0446014, -101.7410278, 76.0186157, -177.8625183, 177.7856140
34: -87.6561890, 58.9865341, -87.5739517, 58.9609718, -146.6171570, 146.5604706
35: -84.8183136, 59.5696526, -84.7230530, 59.5516357, -144.3699493, 144.2927094
36: -79.0712738, 61.3027191, -79.0126038, 61.2785873, -140.3498535, 140.3153229
37: -117.5654907, 65.1638794, -117.5310516, 65.1275635, -182.6930389, 182.6949310
38: -103.4643478, 77.3297501, -103.3806458, 77.3077087, -180.7720642, 180.7103882
39: -118.8428116, 75.6344681, -118.7762604, 75.6185608, -194.4613342, 194.4107361
40: -101.7456970, 62.0066719, -101.6941910, 61.9923897, -163.7380676, 163.7008667
41: -73.3460236, 51.2693024, -73.3206635, 51.2440414, -124.5900497, 124.5899658
42: -56.2494736, 48.2717552, -56.2325859, 48.1840096, -104.4334717, 104.5043335

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1460

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2285201
time: 141.61 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2290699
time: 133.17 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -95.9943085, 66.8632507, -96.2157822, 67.0948334, -163.0891266, 163.0790405
1: -57.5002937, 59.4559517, -57.6547279, 59.6323051, -117.1325989, 117.1106796
2: -49.6100235, 52.0374756, -49.7561417, 52.2934074, -101.9034271, 101.7936096
3: -55.4067268, 62.6856270, -55.5858727, 63.0230751, -118.4298019, 118.2714996
4: -56.9724464, 60.0619850, -57.1420403, 60.3320007, -117.3044434, 117.2040176
5: -57.7222595, 63.5737419, -57.9060936, 63.9122314, -121.6344910, 121.4798355
6: -74.6737366, 51.6589546, -74.8689194, 51.7585831, -126.4323120, 126.5278702
7: -70.2470856, 65.8558197, -70.4609070, 66.0635529, -136.3106384, 136.3167267
8: -68.3751678, 71.6632690, -68.5498047, 71.9542389, -140.3294067, 140.2130737
9: -55.6222267, 60.0906715, -55.8075676, 60.2551193, -115.8773346, 115.8982391
10: -80.4361267, 75.7634354, -80.8262329, 75.8673477, -156.3034668, 156.5896606
11: -87.1694489, 61.9558907, -87.5790634, 62.0502472, -149.2196960, 149.5349426
12: -79.3271866, 69.6164398, -79.7870560, 69.7614822, -149.0886536, 149.4034882
13: -78.9831390, 89.1753769, -79.1352234, 89.4141159, -168.3972473, 168.3106079
14: -122.8999481, 60.4273605, -123.2510071, 60.5241585, -183.4241028, 183.6783447
15: -65.6785355, 58.6195297, -65.8131332, 58.8084946, -124.4870300, 124.4326630
16: -91.8342514, 67.9328003, -92.1208191, 68.0687027, -159.9029541, 160.0536194
17: -127.4352341, 86.1813507, -127.7911911, 86.3053055, -213.7405243, 213.9725342
18: -81.1656494, 65.2139435, -81.4633179, 65.3928528, -146.5585022, 146.6772461
19: -63.4716721, 36.3964348, -63.7155991, 36.5155144, -99.9871597, 100.1120300
20: -55.5541725, 43.9222183, -55.7579117, 44.0166245, -99.5707932, 99.6801300
21: -77.4583740, 44.9103317, -77.7761459, 45.0167694, -122.4751434, 122.6864777
22: -78.4213333, 49.4731445, -78.6231384, 49.6208344, -128.0421753, 128.0962830
23: -63.8600044, 44.5286331, -64.1096649, 44.6887360, -108.5487366, 108.6382904
24: -75.7516479, 42.1682663, -75.9966278, 42.2991371, -118.0507812, 118.1648941
25: -63.9566345, 49.9287186, -64.1539307, 50.0851746, -114.0418091, 114.0826492
26: -90.6384125, 73.3138885, -91.0563965, 73.5249023, -164.1633148, 164.3702850
27: -78.5444489, 49.6091537, -78.7332001, 49.7155991, -128.2600403, 128.3423462
28: -61.6826057, 51.3560181, -61.8807297, 51.5088234, -113.1914215, 113.2367401
29: -84.3907471, 49.1246643, -84.6162720, 49.2297058, -133.6204529, 133.7409363
30: -76.5598145, 54.8250809, -76.7618866, 54.9457092, -131.5055237, 131.5869751
31: -81.5209503, 45.5437469, -81.8437271, 45.6764259, -127.1973724, 127.3874741
32: -70.1307068, 53.2782974, -70.3043976, 53.3525467, -123.4832535, 123.5826874
33: -101.7399597, 75.8880157, -101.8865051, 76.1356125, -177.8755646, 177.7745209
34: -87.5303802, 58.7488289, -87.6963043, 58.9981651, -146.5285339, 146.4451294
35: -84.7246628, 59.3818893, -84.8546753, 59.6651764, -144.3898315, 144.2365723
36: -78.9793854, 61.1234741, -79.1031723, 61.2962494, -140.2756195, 140.2266388
37: -117.3189011, 64.9032898, -117.6001663, 65.0830383, -182.4019470, 182.5034485
38: -103.3263626, 77.1089020, -103.5269241, 77.2972336, -180.6235962, 180.6358185
39: -118.6673050, 75.5044861, -118.8747406, 75.6674271, -194.3347321, 194.3792267
40: -101.5966187, 61.9250374, -101.8056183, 61.9948196, -163.5914307, 163.7306519
41: -73.2163239, 51.1810417, -73.3753357, 51.2751236, -124.4914398, 124.5563812
42: -56.1249619, 48.2430344, -56.3298874, 48.3209457, -104.4459000, 104.5729218

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1460

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2486895
time: 110.53 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2022931, upper bound: 76.2488945
time: 122.97 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -96.2509003, 67.0076065, -96.3239517, 67.1101074, -163.3610077, 163.3315582
1: -57.7327118, 59.5950546, -57.7663460, 59.6475639, -117.3802795, 117.3614044
2: -49.8382263, 52.1735687, -49.8655128, 52.3088455, -102.1470718, 102.0390778
3: -55.6928902, 62.8756065, -55.7232513, 63.0438614, -118.7367477, 118.5988617
4: -57.2312546, 60.2466965, -57.2666702, 60.3569183, -117.5881653, 117.5133667
5: -58.0053596, 63.7720642, -58.0427475, 63.9333763, -121.9387360, 121.8148041
6: -74.8443756, 51.7590179, -74.9313736, 51.7872887, -126.6316605, 126.6903915
7: -70.5431671, 66.0261688, -70.6049347, 66.0794373, -136.6226044, 136.6311035
8: -68.6473846, 71.8360596, -68.6794739, 71.9753418, -140.6227264, 140.5155334
9: -55.8517380, 60.2255173, -55.9134140, 60.2741356, -116.1258698, 116.1389313
10: -80.6409912, 75.8602142, -80.9120789, 75.8986664, -156.5396576, 156.7722931
11: -87.3682556, 62.0898781, -87.6297302, 62.1113129, -149.4795685, 149.7196045
12: -79.4919434, 69.8314819, -79.8177795, 69.8599014, -149.3518372, 149.6492615
13: -79.2139435, 89.3541565, -79.2429504, 89.4469452, -168.6608734, 168.5970764
14: -123.1148605, 60.5592003, -123.3276367, 60.5826187, -183.6974640, 183.8868408
15: -65.8485260, 58.7382774, -65.8917389, 58.8334312, -124.6819611, 124.6300201
16: -92.1093826, 68.0477295, -92.2384109, 68.0855713, -160.1949463, 160.2861328
17: -127.6607895, 86.3269958, -127.8736267, 86.3652802, -214.0260620, 214.2005920
18: -81.3210907, 65.4940948, -81.4925308, 65.5251541, -146.8462524, 146.9866028
19: -63.6288452, 36.5905113, -63.7429504, 36.6108665, -100.2397156, 100.3334656
20: -55.6573601, 44.0653572, -55.7817459, 44.0819664, -99.7393036, 99.8471069
21: -77.6148224, 45.0773277, -77.8149567, 45.0962486, -122.7110748, 122.8922729
22: -78.5919876, 49.6763153, -78.6541443, 49.7162895, -128.3082733, 128.3304596
23: -64.0549011, 44.7890320, -64.1394043, 44.8148956, -108.8697968, 108.9284363
24: -75.9573822, 42.3942947, -76.0215683, 42.4086685, -118.3660507, 118.4158478
25: -64.1221161, 50.1655235, -64.1787872, 50.1992188, -114.3213272, 114.3443146
26: -90.8255692, 73.6328201, -91.0853195, 73.6751709, -164.5007324, 164.7181396
27: -78.7028732, 49.7805710, -78.7615433, 49.7958450, -128.4987183, 128.5421143
28: -61.8478661, 51.5866890, -61.9026070, 51.6192131, -113.4670715, 113.4892960
29: -84.5954819, 49.2888031, -84.6585693, 49.3077240, -133.9031982, 133.9473572
30: -76.7157135, 55.0004044, -76.7936478, 55.0269089, -131.7426147, 131.7940521
31: -81.7278366, 45.7610397, -81.8769073, 45.7818756, -127.5097122, 127.6379395
32: -70.2709045, 53.3583336, -70.3416367, 53.3878136, -123.6587219, 123.6999664
33: -101.8949051, 76.0574417, -101.9319992, 76.2119370, -178.1068420, 177.9894409
34: -87.6961060, 58.9998169, -87.7279892, 59.1193848, -146.8154907, 146.7277832
35: -84.8632202, 59.5798531, -84.8843842, 59.7578812, -144.6210938, 144.4642334
36: -79.0991516, 61.3150902, -79.1272736, 61.3877068, -140.4868469, 140.4423523
37: -117.5849609, 65.1709290, -117.6547852, 65.2101898, -182.7951508, 182.8257141
38: -103.5044632, 77.3410110, -103.5628586, 77.4031067, -180.9075623, 180.9038696
39: -118.8717957, 75.6439667, -118.9277115, 75.7320404, -194.6038208, 194.5716705
40: -101.7742920, 61.9948349, -101.8528900, 62.0256042, -163.7998962, 163.8477173
41: -73.3630676, 51.2686462, -73.4158554, 51.3120956, -124.6751633, 124.6845016
42: -56.2614326, 48.3065147, -56.3771820, 48.3462791, -104.6076965, 104.6837006

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 669
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1330
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1426
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1410
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 668
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 652
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1314
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1489
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1522
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1473
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1521
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1460

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2489904
time: 441.18 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2491906
time: 126.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 570.27 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.1629251, upper bound: 76.1837900
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.2022931, upper bound: 76.1840278
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.1629251, upper bound: 76.1841428
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.1629251, upper bound: 76.1843825
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2039682
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.2022931, upper bound: 76.2041949
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2042811
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2045007
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2285201
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.2022931, upper bound: 76.2287142
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2285201
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2290699
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2486895
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.2022931, upper bound: 76.2488945
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2489904
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 570.27
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2491906

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -95.5948486, 66.5808868, -95.6471939, 66.6601257, -162.2549744, 162.2280884
1: -57.2760086, 59.0304070, -57.3686295, 59.0828552, -116.3588638, 116.3990326
2: -49.3768883, 51.6373177, -49.4211273, 51.7039566, -101.0808411, 101.0584412
3: -55.1631165, 62.1936417, -55.2230492, 62.2535477, -117.4166565, 117.4166870
4: -56.6827850, 59.5487747, -56.7685699, 59.6471596, -116.3299408, 116.3173447
5: -57.4887848, 63.2272110, -57.5407906, 63.3436127, -120.8323975, 120.7679901
6: -74.1117783, 51.4371185, -74.1326599, 51.4928322, -125.6045990, 125.5697784
7: -69.9974823, 65.4624329, -70.0947418, 65.5408173, -135.5382996, 135.5571747
8: -68.0424194, 70.9227448, -68.1620407, 70.9709702, -139.0133972, 139.0847778
9: -55.4685898, 59.8815689, -55.5824928, 59.9132652, -115.3818436, 115.4640656
10: -80.1720428, 75.4459305, -80.2156219, 75.3375549, -155.5095978, 155.6615601
11: -86.6978836, 61.6856880, -86.8325500, 61.6594009, -148.3572845, 148.5182343
12: -78.9181824, 69.2682953, -78.9671326, 69.1969452, -148.1151123, 148.2354279
13: -78.7551880, 88.5912628, -78.8544922, 88.6596222, -167.4147949, 167.4457550
14: -122.5631561, 60.0596542, -122.6639252, 59.9794540, -182.5426025, 182.7235718
15: -65.4413376, 58.2644577, -65.4423676, 58.2689323, -123.7102661, 123.7068176
16: -91.3972549, 67.7190094, -91.4549637, 67.7389679, -159.1362305, 159.1739502
17: -127.1215591, 85.8044434, -127.2250061, 85.6580658, -212.7796021, 213.0294495
18: -80.5750809, 64.8627777, -80.6257324, 65.0143814, -145.5894623, 145.4885101
19: -63.1205406, 36.2418327, -63.1928520, 36.3168983, -99.4374390, 99.4346771
20: -55.3534279, 43.7358246, -55.4080887, 43.7811279, -99.1345520, 99.1439133
21: -77.1260452, 44.7286758, -77.2157745, 44.7571106, -121.8831558, 121.9444427
22: -78.1624603, 49.2980804, -78.2671204, 49.3221130, -127.4845734, 127.5651855
23: -63.5210838, 44.3255539, -63.6726761, 44.4886017, -108.0096893, 107.9982300
24: -75.3642197, 42.0679092, -75.4960251, 42.1907005, -117.5549164, 117.5639191
25: -63.6911774, 49.7781105, -63.7948036, 49.8697243, -113.5608978, 113.5729141
26: -90.2566147, 72.9898376, -90.4355392, 73.0563965, -163.3130188, 163.4253845
27: -78.2561035, 49.4777298, -78.3722992, 49.5611038, -127.8172073, 127.8500214
28: -61.3874817, 51.1539726, -61.5212212, 51.3125305, -112.7000122, 112.6751938
29: -84.0872269, 48.9347763, -84.2419205, 48.9145584, -133.0017853, 133.1766968
30: -76.2570038, 54.5704231, -76.3997803, 54.6450882, -130.9020996, 130.9701996
31: -80.9212875, 45.3536606, -80.9568024, 45.4349060, -126.3561935, 126.3104630
32: -69.7585526, 53.0634460, -69.8156357, 53.0680122, -122.8265457, 122.8790741
33: -101.3999634, 75.7089462, -101.3540268, 75.7964783, -177.1964417, 177.0629730
34: -87.1141815, 58.5313263, -87.1502838, 58.6843948, -145.7985840, 145.6815948
35: -84.3699722, 59.2270050, -84.3671112, 59.3414459, -143.7114258, 143.5941010
36: -78.7183075, 60.9838638, -78.7771149, 61.1024399, -139.8207397, 139.7609863
37: -116.5928268, 64.6470032, -116.7094650, 64.8371964, -181.4300232, 181.3564758
38: -102.9860687, 76.9282684, -103.0146561, 77.0308609, -180.0169373, 179.9429321
39: -118.3260803, 75.3882141, -118.3706818, 75.4325104, -193.7585907, 193.7588959
40: -100.9165649, 61.6648521, -100.9082260, 61.7485008, -162.6650543, 162.5730743
41: -72.7025223, 50.9955826, -72.7651672, 51.0654984, -123.7680206, 123.7607498
42: -55.7931900, 48.0107117, -55.8350449, 47.9922066, -103.7854004, 103.8457565

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1460

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 615

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1506834, upper bound: 76.1530550
time: 119.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1506834, upper bound: 76.1759343
time: 120.09 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -95.7040787, 66.6912079, -95.9341354, 66.8899384, -162.5940247, 162.6253357
1: -57.3063431, 59.2009239, -57.5026741, 59.4402237, -116.7465668, 116.7035980
2: -49.4023056, 51.7876434, -49.5460815, 52.0220337, -101.4243317, 101.3337250
3: -55.1856842, 62.3999023, -55.3596649, 62.6901283, -117.8758087, 117.7595596
4: -56.7086029, 59.7374611, -56.9126167, 60.0467834, -116.7553864, 116.6500778
5: -57.5151443, 63.3599854, -57.6688423, 63.6240883, -121.1392365, 121.0288086
6: -74.3195648, 51.4765587, -74.5766525, 51.6484375, -125.9679947, 126.0531998
7: -70.0240250, 65.6258774, -70.2283020, 65.8811722, -135.9051971, 135.8541870
8: -68.0660553, 71.2054291, -68.3165970, 71.5676117, -139.6336670, 139.5220337
9: -55.5076103, 59.9362755, -55.6801834, 60.0392799, -115.5468903, 115.6164551
10: -80.2629852, 75.4852676, -80.4329681, 75.4740524, -155.7370300, 155.9182281
11: -86.8274536, 61.7282104, -87.1247177, 61.7571030, -148.5845642, 148.8529358
12: -79.0849762, 69.3051758, -79.3170090, 69.3476257, -148.4326019, 148.6221619
13: -78.8013077, 88.7865448, -79.0107193, 89.0787659, -167.8800659, 167.7972717
14: -122.6480713, 60.1519852, -122.8895645, 60.1834412, -182.8315125, 183.0415497
15: -65.4950714, 58.4124718, -65.6268158, 58.5863266, -124.0813980, 124.0392838
16: -91.5707321, 67.7519531, -91.8346939, 67.8549957, -159.4257202, 159.5866394
17: -127.2116089, 85.9256134, -127.4555206, 85.9197311, -213.1313477, 213.3811340
18: -80.7767868, 64.8948135, -81.0642395, 65.1953964, -145.9721832, 145.9590454
19: -63.2423134, 36.2612038, -63.4619370, 36.4009628, -99.6432800, 99.7231445
20: -55.4117584, 43.7764587, -55.5363846, 43.8921814, -99.3039398, 99.3128433
21: -77.2236404, 44.7560120, -77.4366913, 44.8323059, -122.0559235, 122.1926956
22: -78.2361145, 49.3395119, -78.4498978, 49.4126587, -127.6487579, 127.7894135
23: -63.6175690, 44.3573685, -63.8829002, 44.5750351, -108.1925964, 108.2402649
24: -75.4898529, 42.0918045, -75.7778778, 42.2499046, -117.7397537, 117.8696671
25: -63.7810593, 49.8111420, -63.9946327, 49.9495621, -113.7306061, 113.8057709
26: -90.3440399, 73.0403595, -90.6231689, 73.2024612, -163.5464783, 163.6635284
27: -78.3141556, 49.5175934, -78.5106506, 49.6584778, -127.9726334, 128.0282440
28: -61.4668236, 51.1955414, -61.6914368, 51.4165878, -112.8834076, 112.8869781
29: -84.1703568, 48.9793053, -84.4430771, 49.0073471, -133.1777039, 133.4223785
30: -76.3284988, 54.6229477, -76.5618134, 54.7768784, -131.1053772, 131.1847534
31: -81.1656418, 45.3789482, -81.4795532, 45.5571365, -126.7227631, 126.8585052
32: -69.8874817, 53.1018028, -70.0965118, 53.1930351, -123.0805206, 123.1983032
33: -101.5235519, 75.7363205, -101.6207352, 75.9119873, -177.4355316, 177.3570404
34: -87.2436142, 58.5645905, -87.4260864, 58.8123856, -146.0559998, 145.9906769
35: -84.4786835, 59.2535324, -84.5990677, 59.4370918, -143.9157715, 143.8526001
36: -78.7816010, 61.0060692, -78.9095917, 61.1714325, -139.9530334, 139.9156647
37: -116.8456421, 64.6667480, -117.2528915, 64.9816360, -181.8272552, 181.9196472
38: -103.1048813, 76.9693832, -103.2651596, 77.1685104, -180.2733917, 180.2345276
39: -118.4523621, 75.4147339, -118.6464539, 75.5192413, -193.9715881, 194.0611877
40: -101.1682663, 61.6964645, -101.4517822, 61.9309769, -163.0992279, 163.1482544
41: -72.8663635, 51.0266418, -73.1126709, 51.1757469, -124.0421143, 124.1393127
42: -55.9102554, 48.0514717, -56.0838051, 48.1198502, -104.0300980, 104.1352768

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1460

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 615

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1899700, upper bound: 76.1532828
time: 111.50 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1899700, upper bound: 76.1532828
time: 131.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -95.8516388, 66.7251663, -95.7554016, 66.6754150, -162.5270538, 162.4805603
1: -57.5084648, 59.1695671, -57.4802551, 59.0981445, -116.6066132, 116.6498184
2: -49.6051140, 51.7734375, -49.5304451, 51.7194366, -101.3245392, 101.3038788
3: -55.4493217, 62.3836899, -55.3604164, 62.2743492, -117.7236633, 117.7441101
4: -56.9415779, 59.7334747, -56.8931923, 59.6720924, -116.6136703, 116.6266632
5: -57.7719154, 63.4255600, -57.6774635, 63.3647919, -121.1367035, 121.1030121
6: -74.2825928, 51.5369492, -74.1952744, 51.5214729, -125.8040619, 125.7322235
7: -70.2935791, 65.6327744, -70.2387543, 65.5567169, -135.8502960, 135.8715210
8: -68.3146744, 71.0955658, -68.2917633, 70.9921570, -139.3068237, 139.3873138
9: -55.6980553, 60.0163727, -55.6883850, 59.9322052, -115.6302414, 115.7047424
10: -80.3770294, 75.5426025, -80.3016739, 75.3688049, -155.7458191, 155.8442688
11: -86.8966827, 61.8197327, -86.8832855, 61.7205505, -148.6172333, 148.7030182
12: -79.0829926, 69.4833527, -78.9979019, 69.2953949, -148.3783875, 148.4812622
13: -78.9859543, 88.7699585, -78.9621887, 88.6924133, -167.6783600, 167.7321472
14: -122.7778091, 60.1916962, -122.7405014, 60.0379868, -182.8157959, 182.9321899
15: -65.6113205, 58.3831863, -65.5209503, 58.2939911, -123.9053040, 123.9041367
16: -91.6727448, 67.8337860, -91.5728760, 67.7557526, -159.4284973, 159.4066620
17: -127.3468857, 85.9501419, -127.3072891, 85.7180023, -213.0648804, 213.2574310
18: -80.7305450, 65.1429138, -80.6549530, 65.1466599, -145.8771973, 145.7978668
19: -63.2777328, 36.4359016, -63.2202263, 36.4122696, -99.6900024, 99.6561279
20: -55.4566154, 43.8789597, -55.4320374, 43.8464050, -99.3030243, 99.3109894
21: -77.2824707, 44.8956985, -77.2546539, 44.8365707, -122.1190414, 122.1503525
22: -78.3329544, 49.5013084, -78.2979813, 49.4175644, -127.7505188, 127.7992859
23: -63.7159882, 44.5859528, -63.7024765, 44.6147232, -108.3307114, 108.2884140
24: -75.5699844, 42.2939682, -75.5209351, 42.3002777, -117.8702621, 117.8149033
25: -63.8565598, 50.0149574, -63.8195992, 49.9837875, -113.8403473, 113.8345566
26: -90.4437485, 73.3088150, -90.4644699, 73.2065506, -163.6502991, 163.7732697
27: -78.4144440, 49.6491776, -78.4005661, 49.6412964, -128.0557404, 128.0497437
28: -61.5527611, 51.3847008, -61.5430641, 51.4229584, -112.9757156, 112.9277649
29: -84.2918930, 49.0989647, -84.2841492, 48.9926758, -133.2845764, 133.3830872
30: -76.4129562, 54.7457275, -76.4315643, 54.7262802, -131.1392365, 131.1772919
31: -81.1281586, 45.5709763, -80.9901428, 45.5403442, -126.6685028, 126.5611115
32: -69.8987503, 53.1433563, -69.8528748, 53.1032257, -123.0019760, 122.9962311
33: -101.5548706, 75.8783493, -101.3995514, 75.8728485, -177.4277191, 177.2778931
34: -87.2798920, 58.7822647, -87.1819382, 58.8057022, -146.0855865, 145.9642029
35: -84.5085297, 59.4249153, -84.3968506, 59.4341774, -143.9427032, 143.8217468
36: -78.8380737, 61.1755409, -78.8012390, 61.1939087, -140.0319824, 139.9767761
37: -116.8589172, 64.9146576, -116.7641449, 64.9642563, -181.8231812, 181.6788025
38: -103.1642914, 77.1603470, -103.0506592, 77.1366577, -180.3009491, 180.2109985
39: -118.5303726, 75.5276947, -118.4235764, 75.4971161, -194.0274658, 193.9512634
40: -101.0942459, 61.7345924, -100.9554977, 61.7792358, -162.8734741, 162.6900940
41: -72.8492279, 51.0831757, -72.8056946, 51.1024780, -123.9517059, 123.8888702
42: -55.9297142, 48.0740471, -55.8823929, 48.0174103, -103.9471207, 103.9564209

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 669
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1330
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 652
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 1314
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 716
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1489
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1473
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1521
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1460

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 615

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1506834, upper bound: 76.1533833
time: 117.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1506834, upper bound: 76.1762779
time: 150.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 270.31 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 270.31
Output dim: 4, lower bound: -76.1506834, upper bound: 76.1530550
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 270.31
Output dim: 4, lower bound: -76.1506834, upper bound: 76.1759343
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 270.31
Output dim: 4, lower bound: -76.1899700, upper bound: 76.1532828
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 270.31
Output dim: 4, lower bound: -76.1899700, upper bound: 76.1532828
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 270.31
Output dim: 4, lower bound: -76.1506834, upper bound: 76.1533833
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 270.31
Output dim: 4, lower bound: -76.1506834, upper bound: 76.1762779
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 270.31
Output dim: 4, lower bound: -76.1629251, upper bound: 76.1843825
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 270.31
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2039682
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 270.31
Output dim: 4, lower bound: -76.2022931, upper bound: 76.2041949
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 270.31
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2042811
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 270.31
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2045007
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 270.31
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2285201
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 270.31
Output dim: 4, lower bound: -76.2022931, upper bound: 76.2287142
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 270.31
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2285201
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 270.31
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2290699
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 270.31
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2486895
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 270.31
Output dim: 4, lower bound: -76.2022931, upper bound: 76.2488945
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 270.31
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2489904
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 270.31
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2491906

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 122.17 + 7329.58 = 7451.75 seconds
