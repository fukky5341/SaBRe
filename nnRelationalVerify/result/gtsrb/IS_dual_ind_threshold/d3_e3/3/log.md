## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 7200 seconds
Split limit: 100
Threshold: 76.2207571897


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=475, inp2_unstable=475, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

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
execution time: IAR + RelationalAnalysis = 2.91 + 115.61 = 118.52 seconds
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1753

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2646637, upper bound: 76.2199874
time: 966.69 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2646637, upper bound: 76.2646636
time: 123.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1090.15 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1090.15
Output dim: 4, lower bound: -76.2646637, upper bound: 76.2199874
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1090.15
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=474, inp2_unstable=475, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2517139, upper bound: 76.1869157
time: 94.89 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2517139, upper bound: 76.2070318
time: 119.13 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=474, inp2_unstable=475, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

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

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1671

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2517139, upper bound: 76.2316058
time: 118.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2517139, upper bound: 76.2517137
time: 1436.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 1557.89 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1557.89
Output dim: 4, lower bound: -76.2517139, upper bound: 76.1869157
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1557.89
Output dim: 4, lower bound: -76.2517139, upper bound: 76.2070318
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1557.89
Output dim: 4, lower bound: -76.2517139, upper bound: 76.2316058
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1557.89
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=474, inp2_unstable=474, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.47 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -76.2038354, upper bound: 76.1855719
time: 153.24 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -76.2038354, upper bound: 76.1855719
time: 130.98 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=474, inp2_unstable=474, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.46 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2057371
time: 132.43 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2060429
time: 91.95 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=474, inp2_unstable=474, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2302568
time: 154.98 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2306118
time: 123.61 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=474, inp2_unstable=474, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 637

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2504326
time: 328.27 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2507255
time: 145.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 476.49 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 476.49
Output dim: 4, lower bound: -76.2038354, upper bound: 76.1855719
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 476.49
Output dim: 4, lower bound: -76.2038354, upper bound: 76.1855719
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 476.49
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2057371
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 476.49
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2060429
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 476.49
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2302568
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 476.49
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2306118
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 476.49
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2504326
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 476.49
Output dim: 4, lower bound: -76.2038354, upper bound: 76.2507255

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=473, inp2_unstable=474, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.43 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2285201
time: 127.11 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2022931, upper bound: 76.2287142
time: 131.54 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=473, inp2_unstable=474, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.56 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2285201
time: 164.96 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2290699
time: 155.17 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=473, inp2_unstable=474, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.53 seconds

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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2486895
time: 127.66 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2022931, upper bound: 76.2488945
time: 139.04 seconds

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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=473, inp2_unstable=474, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.50 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2489904
time: 427.82 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2491906
time: 124.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 554.54 seconds
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 554.54
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2285201
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 554.54
Output dim: 4, lower bound: -76.2022931, upper bound: 76.2287142
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 554.54
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2285201
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 554.54
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2290699
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 554.54
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2486895
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 554.54
Output dim: 4, lower bound: -76.2022931, upper bound: 76.2488945
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 554.54
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2489904
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 554.54
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2491906

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -95.8422241, 66.7410889, -95.7236786, 66.7339859, -162.5762024, 162.4647675
1: -57.4403267, 59.2761993, -57.3903999, 59.1995735, -116.6399002, 116.6665955
2: -49.5325661, 51.8779640, -49.4399872, 51.8184433, -101.3510132, 101.3179474
3: -55.3280144, 62.4653664, -55.2397804, 62.3813362, -117.7093506, 117.7051468
4: -56.8906898, 59.8619576, -56.7878304, 59.7966499, -116.6873398, 116.6497803
5: -57.6351013, 63.4288521, -57.5588608, 63.4374199, -121.0725250, 120.9877167
6: -74.4457550, 51.6188736, -74.2910461, 51.5246277, -125.9703827, 125.9099197
7: -70.1730652, 65.6813354, -70.1176987, 65.6435165, -135.8165894, 135.7990417
8: -68.2940140, 71.3646240, -68.1814957, 71.1821899, -139.4761963, 139.5461121
9: -55.5707321, 59.9951973, -55.6102867, 59.9590340, -115.5297699, 115.6054764
10: -80.3213806, 75.6212234, -80.2652588, 75.3758392, -155.6972198, 155.8864746
11: -87.0219727, 61.8366623, -86.9774628, 61.6929741, -148.7149506, 148.8141174
12: -79.1466217, 69.4668274, -79.0743408, 69.2292633, -148.3758850, 148.5411682
13: -78.9289398, 88.9517975, -78.8919067, 88.8219452, -167.7508698, 167.8437042
14: -122.7888641, 60.2570877, -122.7288437, 60.0650482, -182.8538971, 182.9859314
15: -65.6101990, 58.4523964, -65.4891357, 58.3481216, -123.9583206, 123.9415207
16: -91.6371460, 67.8668671, -91.5508804, 67.7711182, -159.4082642, 159.4177551
17: -127.3284683, 85.9636307, -127.3013840, 85.7134857, -213.0419617, 213.2650146
18: -80.9430542, 65.1346436, -80.7956848, 65.0390625, -145.9821167, 145.9303284
19: -63.3331795, 36.3519745, -63.2871284, 36.3315468, -99.6647186, 99.6390991
20: -55.4796906, 43.8581467, -55.4599495, 43.8143539, -99.2940445, 99.3181000
21: -77.3439941, 44.8363266, -77.3072586, 44.7778931, -122.1218872, 122.1435776
22: -78.3393097, 49.3861046, -78.3330154, 49.3469124, -127.6862183, 127.7191162
23: -63.7481079, 44.4795914, -63.7742424, 44.5156021, -108.2637100, 108.2538300
24: -75.6088104, 42.1426620, -75.5972748, 42.2090302, -117.8178253, 117.7399292
25: -63.8556633, 49.8707047, -63.8598900, 49.8970299, -113.7526703, 113.7305908
26: -90.5337982, 73.1836166, -90.5607071, 73.0876999, -163.6214905, 163.7443237
27: -78.4577713, 49.5705223, -78.4502563, 49.5855408, -128.0433044, 128.0207825
28: -61.5885353, 51.3069839, -61.6120758, 51.3427124, -112.9312439, 112.9190521
29: -84.2940369, 49.0213699, -84.3148346, 48.9331589, -133.2272034, 133.3362122
30: -76.4761963, 54.7471657, -76.4869308, 54.6852722, -131.1614685, 131.2341003
31: -81.2512512, 45.4933815, -81.1039581, 45.4545364, -126.7057648, 126.5973358
32: -69.9872284, 53.2107735, -69.9141998, 53.1027412, -123.0899658, 123.1249695
33: -101.5623093, 75.8464966, -101.4192047, 75.8226318, -177.3849487, 177.2656860
34: -87.3581085, 58.7008057, -87.2573090, 58.7062950, -146.0643921, 145.9581146
35: -84.5684967, 59.3443146, -84.4536133, 59.3601570, -143.9286499, 143.7979279
36: -78.8860245, 61.0882187, -78.8495178, 61.1153526, -140.0013733, 139.9377441
37: -117.0406418, 64.8758011, -116.9143982, 64.8538666, -181.8944855, 181.7901917
38: -103.1643372, 77.0554962, -103.0849380, 77.0595703, -180.2239075, 180.1404419
39: -118.5081711, 75.4674072, -118.4345474, 75.4642944, -193.9724579, 193.9019470
40: -101.3108292, 61.9040222, -101.0865707, 61.7743988, -163.0852356, 162.9906006
41: -73.0315552, 51.1498833, -72.9210815, 51.0934372, -124.1249771, 124.0709686
42: -55.9932632, 48.1664925, -55.9291611, 48.0263519, -104.0196152, 104.0956573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=473, inp2_unstable=473, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 761
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
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 728
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
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 727
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
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 743
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
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 744
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
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1734
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
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 731
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
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1733
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
type: A, layer: 1, pos: 710
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
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 683
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
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 699
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
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1703
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
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1460

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 615

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -76.1506834, upper bound: 76.1976322
time: 138.76 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -76.1506834, upper bound: 76.2206428
time: 119.98 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -95.9512634, 66.8518829, -96.0101318, 66.9644165, -162.9156799, 162.8620148
1: -57.4705620, 59.4470978, -57.5243416, 59.5574150, -117.0279770, 116.9714355
2: -49.5579071, 52.0285873, -49.5648956, 52.1368942, -101.6948013, 101.5934830
3: -55.3505135, 62.6720390, -55.3763504, 62.8184776, -118.1689911, 118.0483856
4: -56.9163780, 60.0509911, -56.9317741, 60.1966820, -117.1130524, 116.9827652
5: -57.6613274, 63.5619583, -57.6867943, 63.7182846, -121.3796082, 121.2487488
6: -74.6542358, 51.6576385, -74.7356644, 51.6800003, -126.3342361, 126.3933029
7: -70.1996002, 65.8454208, -70.2512207, 65.9845276, -136.1841278, 136.0966492
8: -68.3175812, 71.6477966, -68.3360672, 71.7793732, -140.0969543, 139.9838562
9: -55.6095657, 60.0503616, -55.7079201, 60.0853081, -115.6948700, 115.7582779
10: -80.4123840, 75.6604004, -80.4828644, 75.5123596, -155.9247284, 156.1432648
11: -87.1525269, 61.8786697, -87.2707062, 61.7905846, -148.9431000, 149.1493530
12: -79.3137741, 69.5035553, -79.4246902, 69.3798676, -148.6936340, 148.9282532
13: -78.9746628, 89.1475677, -79.0480881, 89.2415543, -168.2162018, 168.1956329
14: -122.8732147, 60.3502426, -122.9543152, 60.2699432, -183.1431580, 183.3045654
15: -65.6637268, 58.6008034, -65.6732254, 58.6660652, -124.3297882, 124.2740250
16: -91.8113632, 67.8992844, -91.9315643, 67.8870697, -159.6984253, 159.8308411
17: -127.4190140, 86.0849838, -127.5316925, 85.9757996, -213.3948059, 213.6166687
18: -81.1450806, 65.1664352, -81.2345123, 65.2200012, -146.3650818, 146.4009399
19: -63.4552193, 36.3711929, -63.5564537, 36.4155807, -99.8708038, 99.9276428
20: -55.5382462, 43.8984604, -55.5886192, 43.9252090, -99.4634552, 99.4870682
21: -77.4418182, 44.8635674, -77.5285721, 44.8530693, -122.2948914, 122.3921356
22: -78.4128418, 49.4275551, -78.5153503, 49.4377823, -127.8506088, 127.9428864
23: -63.8448753, 44.5110168, -63.9848633, 44.6016159, -108.4464874, 108.4958572
24: -75.7349548, 42.1664200, -75.8798065, 42.2682266, -118.0031815, 118.0462265
25: -63.9456863, 49.9036789, -64.0599213, 49.9767761, -113.9224625, 113.9636002
26: -90.6216888, 73.2335358, -90.7487946, 73.2335052, -163.8551788, 163.9823151
27: -78.5159302, 49.6102600, -78.5886612, 49.6828003, -128.1987305, 128.1989136
28: -61.6680298, 51.3480606, -61.7825317, 51.4465332, -113.1145630, 113.1305923
29: -84.3775864, 49.0657616, -84.5156250, 49.0261421, -133.4037170, 133.5813904
30: -76.5480347, 54.7993965, -76.6493988, 54.8168602, -131.3648987, 131.4487915
31: -81.4962769, 45.5184326, -81.6276855, 45.5766411, -127.0729218, 127.1461182
32: -70.1165009, 53.2486420, -70.1954422, 53.2275238, -123.3440247, 123.4440842
33: -101.6861572, 75.8738632, -101.6863556, 75.9381180, -177.6242523, 177.5602112
34: -87.4877167, 58.7338867, -87.5334549, 58.8341064, -146.3218231, 146.2673340
35: -84.6774750, 59.3707657, -84.6858826, 59.4556961, -144.1331635, 144.0566406
36: -78.9496155, 61.1101723, -78.9824371, 61.1842117, -140.1338196, 140.0925903
37: -117.2940369, 64.8955917, -117.4586716, 64.9983063, -182.2923431, 182.3542633
38: -103.2834778, 77.0961914, -103.3359528, 77.1969452, -180.4804230, 180.4321442
39: -118.6343765, 75.4940262, -118.7107086, 75.5506439, -194.1850281, 194.2047119
40: -101.5630569, 61.9353943, -101.6307449, 61.9567413, -163.5198059, 163.5661316
41: -73.1958389, 51.1806259, -73.2690582, 51.2035522, -124.3993835, 124.4496689
42: -56.1107826, 48.2067871, -56.1783676, 48.1538239, -104.2645950, 104.3851547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=473, inp2_unstable=473, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 761
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
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 745
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
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1721
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
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 744
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
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1699
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
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1691
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
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1692
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
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1490
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
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1761
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
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1393
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
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 835
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
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 607
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
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 699
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
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 804
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
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1460

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 615

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -76.1899700, upper bound: 76.1978102
time: 108.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1899700, upper bound: 76.2208421
time: 130.26 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -96.0989380, 66.8854141, -95.8318710, 66.7492676, -162.8482056, 162.7172699
1: -57.6727715, 59.4153023, -57.5020218, 59.2148895, -116.8876343, 116.9173279
2: -49.7607918, 52.0140686, -49.5493202, 51.8339386, -101.5947113, 101.5633850
3: -55.6141777, 62.6553917, -55.3771477, 62.4021645, -118.0163422, 118.0325394
4: -57.1495132, 60.0466652, -56.9124603, 59.8215752, -116.9710846, 116.9591217
5: -57.9182167, 63.6272125, -57.6955070, 63.4585800, -121.3767853, 121.3227234
6: -74.6164856, 51.7188492, -74.3536224, 51.5532799, -126.1697693, 126.0724640
7: -70.4691467, 65.8516693, -70.2617188, 65.6594391, -136.1285858, 136.1133728
8: -68.5662842, 71.5374146, -68.3112183, 71.2033844, -139.7696533, 139.8486328
9: -55.8002281, 60.1300392, -55.7162094, 59.9779892, -115.7782135, 115.8462372
10: -80.5264053, 75.7179947, -80.3512955, 75.4071121, -155.9335022, 156.0692749
11: -87.2207108, 61.9706726, -87.0281754, 61.7541199, -148.9748230, 148.9988403
12: -79.3114014, 69.6819000, -79.1050949, 69.3277130, -148.6391144, 148.7869873
13: -79.1597290, 89.1304779, -78.9995880, 88.8547058, -168.0144196, 168.1300659
14: -123.0037155, 60.3889389, -122.8054199, 60.1235542, -183.1272736, 183.1943665
15: -65.7801819, 58.5711555, -65.5677643, 58.3732109, -124.1533661, 124.1389084
16: -91.9125137, 67.9817734, -91.6687775, 67.7878876, -159.7004089, 159.6505432
17: -127.5538864, 86.1092987, -127.3837051, 85.7734222, -213.3272858, 213.4929810
18: -81.0984955, 65.4147949, -80.8248901, 65.1713409, -146.2698364, 146.2396851
19: -63.4903488, 36.5460167, -63.3145332, 36.4269180, -99.9172516, 99.8605423
20: -55.5828667, 44.0012970, -55.4838867, 43.8796387, -99.4625092, 99.4851837
21: -77.5004120, 45.0033302, -77.3461456, 44.8573494, -122.3577576, 122.3494720
22: -78.5098648, 49.5893478, -78.3639374, 49.4423828, -127.9522476, 127.9532852
23: -63.9430313, 44.7400246, -63.8040352, 44.6417084, -108.5847321, 108.5440598
24: -75.8145294, 42.3686752, -75.6221771, 42.3185883, -118.1331177, 117.9908371
25: -64.0211182, 50.1075363, -63.8847046, 50.0111008, -114.0322189, 113.9922409
26: -90.7209473, 73.5025787, -90.5896606, 73.2378845, -163.9588318, 164.0922241
27: -78.6161575, 49.7419510, -78.4785004, 49.6657562, -128.2819214, 128.2204590
28: -61.7538338, 51.5376968, -61.6339073, 51.4531479, -113.2069778, 113.1716003
29: -84.4987411, 49.1855545, -84.3570557, 49.0112610, -133.5100098, 133.5426025
30: -76.6320724, 54.9225311, -76.5187302, 54.7664452, -131.3985138, 131.4412537
31: -81.4581299, 45.7106934, -81.1372833, 45.5599823, -127.0181122, 126.8479767
32: -70.1274109, 53.2908020, -69.9514465, 53.1379662, -123.2653732, 123.2422333
33: -101.7172546, 76.0159836, -101.4647217, 75.8990173, -177.6162567, 177.4807129
34: -87.5238647, 58.9517555, -87.2889709, 58.8275604, -146.3514252, 146.2407227
35: -84.7070312, 59.5422668, -84.4834061, 59.4528427, -144.1598816, 144.0256653
36: -79.0058136, 61.2798920, -78.8736420, 61.2068214, -140.2126312, 140.1535339
37: -117.3066940, 65.1434631, -116.9690475, 64.9809723, -182.2876587, 182.1125183
38: -103.3424988, 77.2875824, -103.1209488, 77.1653976, -180.5078735, 180.4085388
39: -118.7126541, 75.6068726, -118.4874573, 75.5288696, -194.2415161, 194.0943298
40: -101.4884796, 61.9737854, -101.1337814, 61.8051682, -163.2936401, 163.1075745
41: -73.1782608, 51.2374916, -72.9616013, 51.1303711, -124.3086319, 124.1990967
42: -56.1297531, 48.2299385, -55.9765015, 48.0515747, -104.1813278, 104.2064285

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=473, inp2_unstable=473, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 653
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
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1751
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
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1590
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
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 727
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
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 744
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
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 697
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
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 852
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
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1736
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
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1708
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
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1426
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1740
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
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1557
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
type: A, layer: 1, pos: 710
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
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1457
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
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 601
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
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1313
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
type: A, layer: 1, pos: 615

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -76.1506834, upper bound: 76.1979443
time: 125.31 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1506834, upper bound: 76.2209928
time: 209.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -96.2078705, 66.9962311, -96.1182632, 66.9797211, -163.1875916, 163.1145020
1: -57.7030029, 59.5862007, -57.6359711, 59.5726891, -117.2756882, 117.2221680
2: -49.7861366, 52.1647110, -49.6742630, 52.1523781, -101.9385071, 101.8389664
3: -55.6366768, 62.8620796, -55.5136909, 62.8392792, -118.4759521, 118.3757706
4: -57.1751938, 60.2356987, -57.0564270, 60.2215996, -117.3967896, 117.2921295
5: -57.9444160, 63.7602921, -57.8234291, 63.7394371, -121.6838531, 121.5837250
6: -74.8248596, 51.7577095, -74.7980957, 51.7087212, -126.5335693, 126.5558014
7: -70.4956665, 66.0157852, -70.3952484, 66.0004425, -136.4960938, 136.4110413
8: -68.5897980, 71.8206253, -68.4657364, 71.8005676, -140.3903656, 140.2863617
9: -55.8390350, 60.1851845, -55.8138237, 60.1043015, -115.9433365, 115.9990005
10: -80.6172485, 75.7571716, -80.5687256, 75.5436249, -156.1608734, 156.3258972
11: -87.3513336, 62.0126038, -87.3213654, 61.8516083, -149.2029419, 149.3339691
12: -79.4785919, 69.7185593, -79.4554901, 69.4782639, -148.9568481, 149.1740417
13: -79.2054520, 89.3263245, -79.1557770, 89.2743378, -168.4797668, 168.4821014
14: -123.0881882, 60.4820633, -123.0310440, 60.3283920, -183.4165802, 183.5130920
15: -65.8337097, 58.7195511, -65.7518845, 58.6910973, -124.5248108, 124.4714355
16: -92.0864792, 68.0142288, -92.0491409, 67.9038849, -159.9903564, 160.0633698
17: -127.6445389, 86.2305984, -127.6142273, 86.0357056, -213.6802368, 213.8448181
18: -81.3005219, 65.4465637, -81.2636948, 65.3522949, -146.6528168, 146.7102661
19: -63.6124420, 36.5652504, -63.5838470, 36.5109406, -100.1233826, 100.1490936
20: -55.6414185, 44.0415993, -55.6125221, 43.9904938, -99.6319122, 99.6541214
21: -77.5982819, 45.0306129, -77.5674133, 44.9325333, -122.5308151, 122.5980148
22: -78.5834656, 49.6307526, -78.5463486, 49.5332031, -128.1166687, 128.1770935
23: -64.0397949, 44.7714272, -64.0146179, 44.7277336, -108.7675171, 108.7860413
24: -75.9406738, 42.3924484, -75.9046936, 42.3777771, -118.3184280, 118.2971420
25: -64.1111908, 50.1404915, -64.0847855, 50.0908470, -114.2020416, 114.2252731
26: -90.8088608, 73.5524597, -90.7778015, 73.3837128, -164.1925659, 164.3302612
27: -78.6743240, 49.7816849, -78.6169586, 49.7630386, -128.4373627, 128.3986511
28: -61.8333244, 51.5787354, -61.8043709, 51.5569534, -113.3902740, 113.3831024
29: -84.5823364, 49.2298660, -84.5579605, 49.1041565, -133.6864929, 133.7878113
30: -76.7039490, 54.9747581, -76.6811523, 54.8980141, -131.6019440, 131.6558838
31: -81.7031860, 45.7357254, -81.6609039, 45.6820869, -127.3852539, 127.3966293
32: -70.2567291, 53.3286781, -70.2327118, 53.2627525, -123.5194702, 123.5613861
33: -101.8410950, 76.0433197, -101.7318802, 76.0144958, -177.8555756, 177.7752075
34: -87.6534882, 58.9848251, -87.5651169, 58.9554291, -146.6089172, 146.5499420
35: -84.8160706, 59.5686874, -84.7156219, 59.5484314, -144.3645020, 144.2843018
36: -79.0694046, 61.3017845, -79.0065384, 61.2756920, -140.3450928, 140.3083191
37: -117.5600586, 65.1631699, -117.5133209, 65.1254120, -182.6854553, 182.6764832
38: -103.4616318, 77.3282394, -103.3719025, 77.3027878, -180.7644043, 180.7001343
39: -118.8389359, 75.6334534, -118.7636032, 75.6152344, -194.4541473, 194.3970642
40: -101.7407608, 62.0051918, -101.6779938, 61.9875412, -163.7283020, 163.6831665
41: -73.3425903, 51.2681732, -73.3095703, 51.2404747, -124.5830536, 124.5777435
42: -56.2472687, 48.2702255, -56.2257004, 48.1790771, -104.4263229, 104.4959259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=473, inp2_unstable=473, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1751
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
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 728
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
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 533
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
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 765
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
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 536
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
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1557
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
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 606
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
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 607
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

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 615

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -76.1899700, upper bound: 76.1981199
time: 137.73 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1506834, upper bound: 76.2211850
time: 116.98 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -95.8822479, 66.7500229, -95.9198608, 66.8563614, -162.7386169, 162.6698914
1: -57.4688034, 59.2816849, -57.5166931, 59.2635345, -116.7323380, 116.7983704
2: -49.5836830, 51.8839340, -49.6280098, 51.9654121, -101.5490952, 101.5119476
3: -55.3834839, 62.4747581, -55.4468803, 62.5722008, -117.9556885, 117.9216385
4: -56.9457436, 59.8693695, -56.9947319, 59.9204407, -116.8661804, 116.8640976
5: -57.6952324, 63.4378090, -57.7755928, 63.6220016, -121.3172302, 121.2134018
6: -74.4612732, 51.6185150, -74.4115753, 51.5978241, -126.0590973, 126.0300903
7: -70.2194138, 65.6881485, -70.3237610, 65.7109146, -135.9303284, 136.0119019
8: -68.3507462, 71.3743362, -68.3924713, 71.3381424, -139.6888733, 139.7668152
9: -55.5819473, 60.0330658, -55.7051620, 60.1211929, -115.7031403, 115.7382202
10: -80.3414764, 75.7230301, -80.5966644, 75.7268906, -156.0683594, 156.3197021
11: -87.0338287, 61.9123993, -87.2693405, 61.9478111, -148.9816437, 149.1817322
12: -79.1560364, 69.5786438, -79.4237213, 69.6073608, -148.7633972, 149.0023651
13: -78.9352646, 88.9776764, -78.9719772, 88.9881897, -167.9234467, 167.9496460
14: -122.8131714, 60.3319740, -123.0179367, 60.3121300, -183.1253052, 183.3499146
15: -65.6225586, 58.4667931, -65.6211472, 58.4761734, -124.0987320, 124.0879364
16: -91.6573105, 67.8989410, -91.7313004, 67.9480743, -159.6053772, 159.6302338
17: -127.3417282, 86.0536346, -127.5509415, 86.0246735, -213.3663635, 213.6045685
18: -80.9596100, 65.1809082, -81.0112228, 65.2079010, -146.1674957, 146.1921082
19: -63.3473015, 36.3765030, -63.4392242, 36.4292831, -99.7765732, 99.8157196
20: -55.4937286, 43.8805809, -55.6229782, 43.9015274, -99.3952484, 99.5035553
21: -77.3578033, 44.8820343, -77.5459137, 44.9381676, -122.2959747, 122.4279404
22: -78.3450470, 49.4264183, -78.4317856, 49.5128517, -127.8578873, 127.8582001
23: -63.7602539, 44.4962273, -63.8895531, 44.5995674, -108.3598099, 108.3857727
24: -75.6219940, 42.1436157, -75.7036438, 42.2371216, -117.8591156, 117.8472595
25: -63.8629456, 49.8944855, -63.9419518, 50.0014267, -113.8643723, 113.8364410
26: -90.5477371, 73.2622528, -90.8592682, 73.3739471, -163.9216614, 164.1215210
27: -78.4847565, 49.5681343, -78.5896149, 49.6143494, -128.0991058, 128.1577454
28: -61.6008873, 51.3137512, -61.7030792, 51.4012146, -113.0021057, 113.0168304
29: -84.3041229, 49.0746002, -84.4054718, 49.1193237, -133.4234467, 133.4800720
30: -76.4861069, 54.7712479, -76.5934143, 54.8091660, -131.2952576, 131.3646545
31: -81.2720947, 45.5173607, -81.3074951, 45.5499458, -126.8220291, 126.8248520
32: -69.9987030, 53.2391510, -70.0143433, 53.2236023, -123.2222900, 123.2534943
33: -101.6133270, 75.8593750, -101.6101990, 76.0159531, -177.6292725, 177.4695587
34: -87.3980865, 58.7140579, -87.4113998, 58.8646698, -146.2627258, 146.1254578
35: -84.6133881, 59.3544617, -84.6150131, 59.5663185, -144.1797028, 143.9694824
36: -78.9139252, 61.1006317, -78.9642029, 61.2244606, -140.1383820, 140.0648193
37: -117.0601120, 64.8828278, -117.0382385, 64.9364700, -181.9965820, 181.9210663
38: -103.2044983, 77.0667114, -103.2673111, 77.1549377, -180.3594208, 180.3340149
39: -118.5372238, 75.4768829, -118.5860825, 75.5777740, -194.1149902, 194.0629578
40: -101.3395233, 61.8921432, -101.2454910, 61.8075981, -163.1471252, 163.1376343
41: -73.0486069, 51.1491776, -73.0163422, 51.1614609, -124.2100601, 124.1655197
42: -56.0052185, 48.2012405, -56.0737839, 48.1885948, -104.1938171, 104.2750168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=473, inp2_unstable=473, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1623
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
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1726
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
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1619
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
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 743
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
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 744
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
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
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
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1736
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
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1734
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
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 645
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
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1733
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
type: A, layer: 1, pos: 710
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
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 700
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
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 769
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
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1703
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
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1460

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 615

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -76.1506834, upper bound: 76.2180325
time: 94.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1506834, upper bound: 76.2407581
time: 116.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -95.9912643, 66.8608246, -96.2060318, 67.0868378, -163.0780945, 163.0668640
1: -57.4990578, 59.4525948, -57.6506271, 59.6213875, -117.1204376, 117.1032257
2: -49.6090126, 52.0345650, -49.7529068, 52.2839127, -101.8929291, 101.7874756
3: -55.4059525, 62.6814384, -55.5834198, 63.0093842, -118.4153290, 118.2648621
4: -56.9713974, 60.0584106, -57.1386337, 60.3204842, -117.2918777, 117.1970444
5: -57.7214241, 63.5708847, -57.9034805, 63.9028778, -121.6242981, 121.4743576
6: -74.6697693, 51.6573029, -74.8561935, 51.7532120, -126.4229813, 126.5134888
7: -70.2459488, 65.8522339, -70.4572754, 66.0519409, -136.2978821, 136.3095093
8: -68.3743057, 71.6575317, -68.5469971, 71.9355316, -140.3098450, 140.2045288
9: -55.6208267, 60.0882111, -55.8029327, 60.2474594, -115.8682709, 115.8911438
10: -80.4324493, 75.7622223, -80.8145065, 75.8633957, -156.2958374, 156.5767212
11: -87.1644135, 61.9543571, -87.5625992, 62.0453453, -149.2097626, 149.5169373
12: -79.3232269, 69.6153412, -79.7741470, 69.7579956, -149.0812225, 149.3894958
13: -78.9809647, 89.1734314, -79.1282120, 89.4078293, -168.3887939, 168.3016357
14: -122.8975830, 60.4251251, -123.2435684, 60.5169258, -183.4145050, 183.6687012
15: -65.6761169, 58.6151810, -65.8051987, 58.7941513, -124.4702682, 124.4203568
16: -91.8314743, 67.9313812, -92.1119080, 68.0640259, -159.8955078, 160.0432892
17: -127.4322281, 86.1749496, -127.7813797, 86.2869186, -213.7191467, 213.9563293
18: -81.1615906, 65.2127228, -81.4500580, 65.3888474, -146.5504150, 146.6627808
19: -63.4693451, 36.3957443, -63.7085838, 36.5132866, -99.9826355, 100.1043243
20: -55.5522614, 43.9209213, -55.7516708, 44.0123444, -99.5646057, 99.6725922
21: -77.4556427, 44.9092712, -77.7672577, 45.0133476, -122.4689941, 122.6765289
22: -78.4185944, 49.4678650, -78.6141052, 49.6036453, -128.0222473, 128.0819702
23: -63.8570023, 44.5276566, -64.1001892, 44.6855087, -108.5425034, 108.6278458
24: -75.7481232, 42.1673698, -75.9861984, 42.2963333, -118.0444412, 118.1535645
25: -63.9529762, 49.9274559, -64.1419830, 50.0811577, -114.0341339, 114.0694427
26: -90.6356583, 73.3122253, -91.0473862, 73.5196533, -164.1553040, 164.3596191
27: -78.5428391, 49.6079025, -78.7280121, 49.7116318, -128.2544556, 128.3359070
28: -61.6803932, 51.3548660, -61.8735123, 51.5051041, -113.1854858, 113.2283707
29: -84.3876801, 49.1190033, -84.6062775, 49.2122917, -133.5999756, 133.7252808
30: -76.5579681, 54.8234634, -76.7558746, 54.9406090, -131.4985657, 131.5793457
31: -81.5171204, 45.5423927, -81.8313675, 45.6720810, -127.1892014, 127.3737640
32: -70.1279907, 53.2770042, -70.2956390, 53.3484230, -123.4764099, 123.5726471
33: -101.7371521, 75.8867340, -101.8773422, 76.1314545, -177.8686066, 177.7640686
34: -87.5276642, 58.7471466, -87.6874619, 58.9925842, -146.5202484, 146.4346008
35: -84.7223511, 59.3808861, -84.8472443, 59.6619873, -144.3843384, 144.2281342
36: -78.9775009, 61.1226006, -79.0971222, 61.2933655, -140.2708740, 140.2197113
37: -117.3134308, 64.9026031, -117.5824127, 65.0808868, -182.3943176, 182.4850159
38: -103.3236084, 77.1074371, -103.5181808, 77.2923431, -180.6159515, 180.6256104
39: -118.6633911, 75.5034485, -118.8621521, 75.6641388, -194.3275146, 194.3656006
40: -101.5916519, 61.9235611, -101.7894516, 61.9899559, -163.5816040, 163.7130127
41: -73.2128906, 51.1799316, -73.3642578, 51.2716103, -124.4844971, 124.5441895
42: -56.1227417, 48.2414970, -56.3229752, 48.3160248, -104.4387512, 104.5644684

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=473, inp2_unstable=473, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=671, inp2_unstable=671, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=30, inp2_unstable=30, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1747
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
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1726
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
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1619
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
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1721
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
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1743
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
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1699
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
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1791
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
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1546
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
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1490
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
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1410
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 668
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 645
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
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1393
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
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1522
type: A, layer: 1, pos: 835
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
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 607
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
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 804
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
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1460

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 615

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -76.1899700, upper bound: 76.2182051
time: 143.95 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1899700, upper bound: 76.2409619
time: 2936.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3083.20 seconds
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3083.20
Output dim: 4, lower bound: -76.1506834, upper bound: 76.1976322
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3083.20
Output dim: 4, lower bound: -76.1506834, upper bound: 76.2206428
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3083.20
Output dim: 4, lower bound: -76.1899700, upper bound: 76.1978102
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3083.20
Output dim: 4, lower bound: -76.1899700, upper bound: 76.2208421
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3083.20
Output dim: 4, lower bound: -76.1506834, upper bound: 76.1979443
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3083.20
Output dim: 4, lower bound: -76.1506834, upper bound: 76.2209928
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3083.20
Output dim: 4, lower bound: -76.1899700, upper bound: 76.1981199
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3083.20
Output dim: 4, lower bound: -76.1506834, upper bound: 76.2211850
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3083.20
Output dim: 4, lower bound: -76.1506834, upper bound: 76.2180325
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3083.20
Output dim: 4, lower bound: -76.1506834, upper bound: 76.2407581
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3083.20
Output dim: 4, lower bound: -76.1899700, upper bound: 76.2182051
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3083.20
Output dim: 4, lower bound: -76.1899700, upper bound: 76.2409619
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3083.20
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2489904
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3083.20
Output dim: 4, lower bound: -76.1629251, upper bound: 76.2491906

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 118.52 + 9937.60 = 10056.12 seconds
