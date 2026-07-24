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
execution time: IAR + RelationalAnalysis = 2.82 + 120.81 = 123.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -76.2665171, upper bound: 76.2665171

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2428148, upper bound: 76.2600672
time: 107.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2600672, upper bound: 76.2428148
time: 115.06 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 222.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 222.72
Output dim: 4, lower bound: -76.2428148, upper bound: 76.2600672
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 222.72
Output dim: 4, lower bound: -76.2600672, upper bound: 76.2428148

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2167893, upper bound: 76.2473206
time: 196.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2308354, upper bound: 76.2343750
time: 251.14 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2343750, upper bound: 76.2308354
time: 153.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2473206, upper bound: 76.2167893
time: 147.43 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 303.62 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 303.62
Output dim: 4, lower bound: -76.2167893, upper bound: 76.2473206
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 303.62
Output dim: 4, lower bound: -76.2308354, upper bound: 76.2343750
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 303.62
Output dim: 4, lower bound: -76.2343750, upper bound: 76.2308354
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 303.62
Output dim: 4, lower bound: -76.2473206, upper bound: 76.2167893

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1753

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2149428, upper bound: 76.2009650
time: 131.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1703414, upper bound: 76.2454749
time: 107.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1753

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2289882, upper bound: 76.1880205
time: 209.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1844213, upper bound: 76.2325331
time: 232.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1753

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2325331, upper bound: 76.1844213
time: 118.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1880205, upper bound: 76.2289882
time: 138.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1753

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2454749, upper bound: 76.1703414
time: 132.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2009650, upper bound: 76.2149428
time: 131.89 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 266.91 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 266.91
Output dim: 4, lower bound: -76.2149428, upper bound: 76.2009650
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 266.91
Output dim: 4, lower bound: -76.1703414, upper bound: 76.2454749
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 266.91
Output dim: 4, lower bound: -76.2289882, upper bound: 76.1880205
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 266.91
Output dim: 4, lower bound: -76.1844213, upper bound: 76.2325331
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 266.91
Output dim: 4, lower bound: -76.2325331, upper bound: 76.1844213
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 266.91
Output dim: 4, lower bound: -76.1880205, upper bound: 76.2289882
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 266.91
Output dim: 4, lower bound: -76.2454749, upper bound: 76.1703414
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 266.91
Output dim: 4, lower bound: -76.2009650, upper bound: 76.2149428

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1791411, upper bound: 76.1719745
time: 177.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1881681, upper bound: 76.1609131
time: 120.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1346212, upper bound: 76.2164244
time: 104.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1436716, upper bound: 76.2053785
time: 124.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1885456, upper bound: 76.1618128
time: 148.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1987472, upper bound: 76.1513351
time: 111.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1440512, upper bound: 76.2062431
time: 91.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1542592, upper bound: 76.1957740
time: 107.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1957740, upper bound: 76.1542592
time: 111.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2062431, upper bound: 76.1440512
time: 133.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1513351, upper bound: 76.1987472
time: 134.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1618128, upper bound: 76.1885456
time: 122.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2053785, upper bound: 76.1436716
time: 106.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.2164244, upper bound: 76.1346212
time: 116.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1656

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1609131, upper bound: 76.1881681
time: 147.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1719745, upper bound: 76.1791411
time: 133.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 283.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.1791411, upper bound: 76.1719745
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.1881681, upper bound: 76.1609131
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.1346212, upper bound: 76.2164244
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.1436716, upper bound: 76.2053785
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.1885456, upper bound: 76.1618128
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.1987472, upper bound: 76.1513351
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.1440512, upper bound: 76.2062431
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.1542592, upper bound: 76.1957740
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.1957740, upper bound: 76.1542592
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.2062431, upper bound: 76.1440512
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.1513351, upper bound: 76.1987472
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.1618128, upper bound: 76.1885456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.2053785, upper bound: 76.1436716
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.2164244, upper bound: 76.1346212
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.1609131, upper bound: 76.1881681
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 283.52
Output dim: 4, lower bound: -76.1719745, upper bound: 76.1791411

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1775984, upper bound: 76.1344624
time: 114.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1413495, upper bound: 76.1704375
time: 102.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1866264, upper bound: 76.1233385
time: 126.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1504167, upper bound: 76.1593728
time: 1104.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1330781, upper bound: 76.1788877
time: 175.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.0968073, upper bound: 76.2148941
time: 98.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1421293, upper bound: 76.1677845
time: 114.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1058985, upper bound: 76.2038403
time: 126.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1751

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1870076, upper bound: 76.1241796
time: 316.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.0968073, upper bound: 76.1602757
time: 113.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1751

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1972091, upper bound: 76.1135899
time: 128.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1610980, upper bound: 76.1497966
time: 112.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -96.3063354, 67.0364227, -96.3063354, 67.0364227, -163.3427429, 163.3427429
1: -57.7626457, 59.6254387, -57.7626457, 59.6254387, -117.3880692, 117.3880768
2: -49.8706818, 52.2028503, -49.8706818, 52.2028503, -102.0735168, 102.0735168
3: -55.7297554, 62.9107666, -55.7297554, 62.9107666, -118.6405182, 118.6405182
4: -57.2672997, 60.2849045, -57.2672997, 60.2849045, -117.5522003, 117.5522003
5: -58.0393372, 63.8028450, -58.0393372, 63.8028450, -121.8421783, 121.8421783
6: -74.8995972, 51.8062744, -74.8995972, 51.8062744, -126.7058716, 126.7058716
7: -70.5811691, 66.0594330, -70.5811691, 66.0594330, -136.6405945, 136.6405945
8: -68.6855927, 71.8806381, -68.6855927, 71.8806381, -140.5662231, 140.5662231
9: -55.8761787, 60.2607040, -55.8761787, 60.2607040, -116.1368790, 116.1368713
10: -80.6720276, 75.9325104, -80.6720276, 75.9325104, -156.6045380, 156.6045380
11: -87.4131775, 62.1444702, -87.4131775, 62.1444702, -149.5576477, 149.5576477
12: -79.5313721, 69.8858643, -79.5313721, 69.8858643, -149.4172363, 149.4172363
13: -79.2643509, 89.4011841, -79.2643509, 89.4011841, -168.6655273, 168.6655273
14: -123.1595383, 60.6075783, -123.1595383, 60.6075783, -183.7670898, 183.7671204
15: -65.9118271, 58.7847443, -65.9118271, 58.7847443, -124.6965714, 124.6965714
16: -92.1803589, 68.1027374, -92.1803589, 68.1027374, -160.2830963, 160.2830963
17: -127.7027664, 86.3878403, -127.7027664, 86.3878403, -214.0906067, 214.0906067
18: -81.3621979, 65.5256805, -81.3621979, 65.5256805, -146.8878632, 146.8878784
19: -63.6586571, 36.6148415, -63.6586571, 36.6148415, -100.2734985, 100.2734985
20: -55.6837273, 44.0900726, -55.6837273, 44.0900726, -99.7737885, 99.7738037
21: -77.6484833, 45.1083336, -77.6484833, 45.1083336, -122.7568207, 122.7568207
22: -78.6387939, 49.7083244, -78.6387939, 49.7083244, -128.3471069, 128.3471069
23: -64.0853577, 44.8193665, -64.0853577, 44.8193665, -108.9047089, 108.9047089
24: -75.9946899, 42.4198761, -75.9946899, 42.4198761, -118.4145584, 118.4145660
25: -64.1596985, 50.1932678, -64.1596985, 50.1932678, -114.3529663, 114.3529663
26: -90.8695068, 73.6821594, -90.8695068, 73.6821594, -164.5516510, 164.5516663
27: -78.7443237, 49.8088455, -78.7443237, 49.8088455, -128.5531616, 128.5531616
28: -61.8784943, 51.6111679, -61.8784943, 51.6111679, -113.4896622, 113.4896622
29: -84.6347046, 49.3209190, -84.6347046, 49.3209190, -133.9556274, 133.9556274
30: -76.7502747, 55.0431137, -76.7502747, 55.0431137, -131.7933655, 131.7933807
31: -81.7687225, 45.7890129, -81.7687225, 45.7890129, -127.5577316, 127.5577393
32: -70.3095703, 53.3889275, -70.3095703, 53.3889275, -123.6984940, 123.6985016
33: -101.9429932, 76.0823364, -101.9429932, 76.0823364, -178.0253296, 178.0253296
34: -87.7384720, 59.0245743, -87.7384720, 59.0245743, -146.7630310, 146.7630157
35: -84.9043045, 59.5992126, -84.9043045, 59.5992126, -144.5035095, 144.5035095
36: -79.1380920, 61.3314972, -79.1380920, 61.3314972, -140.4695740, 140.4695740
37: -117.6478271, 65.2042923, -117.6478271, 65.2042923, -182.8521118, 182.8521118
38: -103.5543365, 77.3655243, -103.5543365, 77.3655243, -180.9198303, 180.9198456
39: -118.9177094, 75.6682510, -118.9177094, 75.6682510, -194.5859528, 194.5859528
40: -101.8334045, 62.0376282, -101.8334045, 62.0376282, -163.8710327, 163.8710327
41: -73.4092865, 51.3211327, -73.4092865, 51.3211327, -124.7304230, 124.7304230
42: -56.2979355, 48.3570099, -56.2979355, 48.3570099, -104.6549301, 104.6549377

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 785

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1751

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1425109, upper bound: 76.1686190
time: 125.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -76.1063856, upper bound: 76.2047119
time: 144.43 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 272.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.1775984, upper bound: 76.1344624
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.1413495, upper bound: 76.1704375
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.1866264, upper bound: 76.1233385
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.1504167, upper bound: 76.1593728
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.1330781, upper bound: 76.1788877
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.0968073, upper bound: 76.2148941
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.1421293, upper bound: 76.1677845
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.1058985, upper bound: 76.2038403
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.1870076, upper bound: 76.1241796
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.0968073, upper bound: 76.1602757
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.1972091, upper bound: 76.1135899
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.1610980, upper bound: 76.1497966
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.1425109, upper bound: 76.1686190
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 272.36
Output dim: 4, lower bound: -76.1063856, upper bound: 76.2047119
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 272.36
Output dim: 4, lower bound: -76.1542592, upper bound: 76.1957740
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 272.36
Output dim: 4, lower bound: -76.1957740, upper bound: 76.1542592
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 272.36
Output dim: 4, lower bound: -76.2062431, upper bound: 76.1440512
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 272.36
Output dim: 4, lower bound: -76.1513351, upper bound: 76.1987472
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 272.36
Output dim: 4, lower bound: -76.1618128, upper bound: 76.1885456
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 272.36
Output dim: 4, lower bound: -76.2053785, upper bound: 76.1436716
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 272.36
Output dim: 4, lower bound: -76.2164244, upper bound: 76.1346212
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 272.36
Output dim: 4, lower bound: -76.1609131, upper bound: 76.1881681
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 272.36
Output dim: 4, lower bound: -76.1719745, upper bound: 76.1791411

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 123.63 + 7120.89 = 7244.52 seconds
