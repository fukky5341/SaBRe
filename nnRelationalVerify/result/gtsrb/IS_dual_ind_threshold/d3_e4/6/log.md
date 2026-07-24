## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 6)
Time budget: 7200 seconds
Split limit: 100
Threshold: 84.4938230985


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076)
1: (-57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798)
2: (-49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902)
3: (-62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524)
4: (-64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108)
5: (-59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546)
6: (-94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902)
7: (-66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507)
8: (-81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829)
9: (-61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657)
10: (-88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903)
11: (-86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729)
12: (-97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203)
13: (-85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262)
14: (-144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866)
15: (-78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936)
16: (-91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036)
17: (-133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780)
18: (-93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042)
19: (-67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516)
20: (-68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736)
21: (-85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806)
22: (-86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216)
23: (-70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971)
24: (-90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506)
25: (-76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908)
26: (-101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638)
27: (-88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519)
28: (-68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854)
29: (-89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516)
30: (-88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031)
31: (-91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580)
32: (-90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312)
33: (-127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043)
34: (-106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450)
35: (-99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106)
36: (-92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861)
37: (-145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073)
38: (-112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213)
39: (-133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469)
40: (-111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833)
41: (-96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343)
42: (-70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.94 + 126.96 = 129.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -84.5784015, upper bound: 84.5784015

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1685

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5727068, upper bound: 84.4922254
time: 513.91 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5727068, upper bound: 84.5727067
time: 95.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 609.67 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 609.67
Output dim: 9, lower bound: -84.5727068, upper bound: 84.4922254
IS_A2, status: Status.UNKNOWN, split count: 1, time: 609.67
Output dim: 9, lower bound: -84.5727068, upper bound: 84.5727067

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -108.6825104, 78.8020706, -108.8526688, 78.8883667, -187.5708771, 187.6547241
1: -57.0678215, 59.0297890, -57.1739540, 59.0939026, -116.1617279, 116.2037430
2: -49.3603210, 60.4166794, -49.4335709, 60.4860802, -109.8463898, 109.8502502
3: -62.0270157, 73.3680878, -62.1164703, 73.4865723, -135.5135803, 135.4845581
4: -64.6978760, 70.6075897, -64.7954407, 70.7219849, -135.4198608, 135.4030151
5: -59.6082916, 73.0030823, -59.7284241, 73.1180878, -132.7263794, 132.7315063
6: -94.2190552, 62.8569450, -94.3569489, 62.9529648, -157.1720276, 157.2138824
7: -66.5231934, 69.4289551, -66.6753235, 69.5176086, -136.0407867, 136.1042633
8: -81.3496399, 83.7510376, -81.4347305, 83.8539886, -165.2036133, 165.1857605
9: -60.6766510, 76.6171570, -60.9696350, 76.7890930, -137.4657440, 137.5867920
10: -88.3650818, 90.9195251, -88.8624115, 91.1868896, -179.5519714, 179.7819366
11: -85.7005463, 58.0707436, -86.0296936, 58.1721954, -143.8727417, 144.1004333
12: -97.5419464, 77.0192108, -97.6799011, 77.1191254, -174.6610413, 174.6991119
13: -84.9839172, 98.4149780, -85.0963058, 98.5395813, -183.5234985, 183.5112915
14: -144.1477051, 82.0576172, -144.4921570, 82.2320404, -226.3797302, 226.5497742
15: -78.6271057, 64.2450562, -78.7141647, 64.3557434, -142.9828491, 142.9592285
16: -90.9301682, 72.2346725, -91.3185730, 72.4116592, -163.3418274, 163.5532532
17: -133.2652130, 71.6172714, -133.5927124, 71.7686920, -205.0338745, 205.2099915
18: -93.2614899, 69.7886658, -93.4120255, 69.8717575, -163.1332397, 163.2006836
19: -67.6487885, 40.3830147, -67.7763977, 40.4224014, -108.0711899, 108.1594086
20: -68.4995728, 53.0521393, -68.6140442, 53.1098595, -121.6094360, 121.6661835
21: -85.0168381, 51.1076126, -85.2239761, 51.1771736, -136.1940155, 136.3315887
22: -86.4652023, 46.3651047, -86.6394806, 46.4932442, -132.9584351, 133.0045776
23: -69.9554596, 54.0113220, -70.0875702, 54.0643234, -124.0197830, 124.0988846
24: -90.2603989, 54.3981781, -90.3793106, 54.4904404, -144.7508392, 144.7774963
25: -76.0829163, 55.4306641, -76.1795883, 55.5280952, -131.6110077, 131.6102600
26: -100.8495712, 82.0273743, -101.0005188, 82.1109924, -182.9605713, 183.0278931
27: -87.7463531, 49.5539932, -87.8881912, 49.6350250, -137.3813782, 137.4421844
28: -68.4534607, 54.4348679, -68.5318451, 54.4931297, -122.9465790, 122.9667130
29: -89.0571289, 41.7203522, -89.2190628, 41.7782173, -130.8353271, 130.9394226
30: -88.2577209, 63.5894547, -88.4003906, 63.6917305, -151.9494171, 151.9898376
31: -91.4716110, 56.0199738, -91.6042023, 56.0578461, -147.5294495, 147.6241760
32: -89.9175873, 57.4322014, -90.0403061, 57.5081940, -147.4257812, 147.4725037
33: -126.6107407, 77.8046570, -126.7966995, 78.1587143, -204.7694550, 204.6013336
34: -106.3053589, 48.5926857, -106.4135208, 48.7997093, -155.1050720, 155.0061951
35: -99.0683594, 58.6328049, -99.2008438, 58.9401741, -158.0085297, 157.8336487
36: -92.4493256, 57.0919380, -92.5959167, 57.3570251, -149.8063507, 149.6878357
37: -145.2531738, 62.5798264, -145.5114746, 62.8878860, -208.1410522, 208.0913086
38: -112.1943970, 71.0787964, -112.3876038, 71.4290237, -183.6234131, 183.4664001
39: -133.0400085, 76.3493347, -133.2745056, 76.6960220, -209.7360229, 209.6238403
40: -110.8761749, 56.7141113, -111.0649490, 56.9452133, -167.8213806, 167.7790527
41: -95.7493744, 65.7828827, -95.9017944, 65.9326248, -161.6820068, 161.6846619
42: -70.2251740, 56.6016235, -70.3509369, 56.6778984, -126.9030685, 126.9525604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=522, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1036

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1669

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5079515, upper bound: 84.4868111
time: 98.42 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5704435, upper bound: 84.4868111
time: 118.76 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -108.9021606, 78.9381256, -108.9073181, 78.9452515, -187.8473969, 187.8454437
1: -57.1944237, 59.1250954, -57.1964493, 59.1272202, -116.3216400, 116.3215485
2: -49.4602089, 60.5124817, -49.4649239, 60.5145721, -109.9747772, 109.9774017
3: -62.1488037, 73.5440979, -62.1527672, 73.5565262, -135.7053223, 135.6968689
4: -64.8313599, 70.7698593, -64.8429337, 70.7730255, -135.6043854, 135.6127777
5: -59.7581177, 73.1889801, -59.7612534, 73.1978455, -132.9559631, 132.9502258
6: -94.4425659, 62.9719315, -94.4505463, 62.9736900, -157.4162445, 157.4224548
7: -66.7029877, 69.5845032, -66.7052917, 69.5897598, -136.2927551, 136.2897949
8: -81.4612885, 83.8939972, -81.4636688, 83.8969803, -165.3582611, 165.3576660
9: -60.9977417, 76.9633636, -61.0006142, 76.9712677, -137.9689941, 137.9639740
10: -88.8991547, 91.4643250, -88.9022293, 91.4784088, -180.3775635, 180.3665466
11: -86.0711594, 58.2591438, -86.0738831, 58.2652740, -144.3364258, 144.3330231
12: -97.7410355, 77.1502075, -97.7449188, 77.1533356, -174.8943787, 174.8951263
13: -85.1333160, 98.5814056, -85.1384125, 98.5847092, -183.7180176, 183.7197876
14: -144.5381165, 82.4331665, -144.5428162, 82.4405212, -226.9786377, 226.9759827
15: -78.7409515, 64.4085541, -78.7442017, 64.4118881, -143.1528320, 143.1527557
16: -91.3679581, 72.5947571, -91.3721008, 72.6038361, -163.9717865, 163.9668579
17: -133.6418152, 71.8285294, -133.6455231, 71.8322983, -205.4741058, 205.4740601
18: -93.4431305, 69.9135056, -93.4593124, 69.9179840, -163.3610840, 163.3728180
19: -67.8218231, 40.4408417, -67.8257446, 40.4433060, -108.2651291, 108.2665863
20: -68.6486969, 53.1403923, -68.6514664, 53.1435356, -121.7922363, 121.7918549
21: -85.2771988, 51.2163391, -85.2808075, 51.2211227, -136.4983215, 136.4971466
22: -86.7549820, 46.5233116, -86.7611465, 46.5258026, -133.2807922, 133.2844543
23: -70.1270905, 54.0933571, -70.1302185, 54.0970993, -124.2241745, 124.2235641
24: -90.4580688, 54.5093765, -90.4725113, 54.5128021, -144.9708557, 144.9818726
25: -76.2221527, 55.5569839, -76.2249985, 55.5598145, -131.7819519, 131.7819824
26: -101.0425262, 82.1481018, -101.0569839, 82.1527710, -183.1952972, 183.2050781
27: -87.9699097, 49.6536369, -87.9849701, 49.6570740, -137.6269836, 137.6386108
28: -68.5663452, 54.5140762, -68.5705490, 54.5162621, -123.0826111, 123.0846252
29: -89.2900162, 41.8007965, -89.2954712, 41.8031998, -131.0932159, 131.0962677
30: -88.4397125, 63.7614975, -88.4428482, 63.7657280, -152.2054291, 152.2043457
31: -91.6505814, 56.0764847, -91.6592941, 56.0800934, -147.7306824, 147.7357788
32: -90.1213760, 57.5232277, -90.1286087, 57.5247307, -147.6461029, 147.6518402
33: -126.9933014, 78.1782227, -127.0036545, 78.1802521, -205.1735535, 205.1818542
34: -106.5051193, 48.8233833, -106.5143280, 48.8250656, -155.3301544, 155.3377075
35: -99.3263855, 58.9572334, -99.3339081, 58.9585266, -158.2848816, 158.2911377
36: -92.7336502, 57.3715973, -92.7420425, 57.3724747, -150.1061249, 150.1136475
37: -145.7526855, 62.9080963, -145.7646179, 62.9101486, -208.6628418, 208.6727142
38: -112.5565109, 71.4561234, -112.5665970, 71.4584808, -184.0149841, 184.0227051
39: -133.4869385, 76.7122040, -133.4977264, 76.7143250, -210.2012634, 210.2099304
40: -111.2218399, 56.9596291, -111.2300339, 56.9609604, -168.1828003, 168.1896667
41: -96.0328751, 65.9454193, -96.0405121, 65.9472122, -161.9800873, 161.9859314
42: -70.4235611, 56.6825485, -70.4280777, 56.6937981, -127.1173553, 127.1106262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=522, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1036

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1669

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5079515, upper bound: 84.5704434
time: 96.97 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5704435, upper bound: 84.5704434
time: 115.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 214.75 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 214.75
Output dim: 9, lower bound: -84.5079515, upper bound: 84.4868111
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 214.75
Output dim: 9, lower bound: -84.5704435, upper bound: 84.4868111
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 214.75
Output dim: 9, lower bound: -84.5079515, upper bound: 84.5704434
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 214.75
Output dim: 9, lower bound: -84.5704435, upper bound: 84.5704434

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -108.6450577, 78.7801285, -108.7199097, 78.8095856, -187.4546356, 187.5000305
1: -57.0437813, 59.0092201, -57.0908585, 59.0085793, -116.0523605, 116.1000824
2: -49.3082695, 60.3948326, -49.3125114, 60.3760643, -109.6843338, 109.7073364
3: -61.9497185, 73.3426666, -61.9459877, 73.3279877, -135.2777100, 135.2886505
4: -64.6529312, 70.5813217, -64.6748581, 70.5908966, -135.2438354, 135.2561646
5: -59.5558052, 72.9783020, -59.6005058, 72.9874268, -132.5432281, 132.5788116
6: -94.1143951, 62.8400879, -94.1173859, 62.8409500, -156.9553528, 156.9574585
7: -66.4691315, 69.4072037, -66.5257111, 69.4084167, -135.8775330, 135.9329071
8: -81.3145142, 83.7252731, -81.3289642, 83.7343750, -165.0488586, 165.0542297
9: -60.6495209, 76.4946289, -60.7209167, 76.5563278, -137.2058411, 137.2155457
10: -88.3303680, 90.6665497, -88.3642883, 90.7165070, -179.0468750, 179.0308380
11: -85.6710434, 57.9992714, -85.7822723, 58.0259018, -143.6969452, 143.7815399
12: -97.5068207, 76.9095917, -97.3722382, 76.8959503, -174.4027710, 174.2818146
13: -84.9413757, 98.3753433, -84.9309235, 98.4077682, -183.3491364, 183.3062744
14: -144.1000061, 81.8773956, -144.0882263, 81.8976746, -225.9976807, 225.9656219
15: -78.5911102, 64.1916351, -78.5846024, 64.2051086, -142.7962036, 142.7762299
16: -90.8874817, 72.1186981, -91.0403366, 72.1770020, -163.0644836, 163.1590271
17: -133.2335205, 71.5600281, -133.3401642, 71.6209412, -204.8544617, 204.9001923
18: -93.2232590, 69.7450256, -93.2542343, 69.7385788, -162.9618225, 162.9992523
19: -67.6138458, 40.3517380, -67.6251526, 40.3381882, -107.9520340, 107.9768829
20: -68.4691086, 53.0381317, -68.4958344, 53.0558014, -121.5248947, 121.5339661
21: -84.9778748, 51.0550232, -85.0136642, 51.0550880, -136.0329590, 136.0686951
22: -86.4105072, 46.3307915, -86.4756622, 46.3917732, -132.8022766, 132.8064575
23: -69.9276581, 53.9842300, -69.9639587, 53.9868202, -123.9144592, 123.9481888
24: -90.2111816, 54.3800278, -90.2571335, 54.3868675, -144.5980530, 144.6371613
25: -76.0539551, 55.4034157, -76.0709839, 55.4276352, -131.4815979, 131.4743958
26: -100.8124313, 81.9349060, -100.7893448, 81.9033966, -182.7158203, 182.7242279
27: -87.6508331, 49.5376472, -87.6716232, 49.5117264, -137.1625519, 137.2092743
28: -68.3938599, 54.4156036, -68.3895721, 54.3747253, -122.7685852, 122.8051758
29: -89.0108719, 41.6729355, -89.0683670, 41.6758118, -130.6866760, 130.7413025
30: -88.2245255, 63.5557861, -88.2842484, 63.5961647, -151.8206940, 151.8400269
31: -91.4300919, 55.9904823, -91.4468460, 55.9760551, -147.4061432, 147.4373169
32: -89.8564453, 57.4194946, -89.8802185, 57.4432449, -147.2996826, 147.2997131
33: -126.4794083, 77.7802582, -126.5437775, 77.8627243, -204.3421173, 204.3240204
34: -106.1762085, 48.5699997, -106.1586380, 48.5608902, -154.7370911, 154.7286377
35: -98.9516602, 58.6153984, -98.9733429, 58.6589737, -157.6106262, 157.5887451
36: -92.3305893, 57.0804367, -92.3498688, 57.1727486, -149.5033417, 149.4302979
37: -145.1668091, 62.5547333, -145.2961884, 62.7108841, -207.8776855, 207.8509216
38: -112.0493469, 71.0532455, -112.0887299, 71.1620712, -183.2114258, 183.1419678
39: -132.9547882, 76.3295288, -133.0614319, 76.5018158, -209.4565735, 209.3909607
40: -110.7793198, 56.7019386, -110.8458710, 56.7845688, -167.5638885, 167.5478058
41: -95.6619568, 65.7656250, -95.7013779, 65.8102722, -161.4722290, 161.4669952
42: -70.1752472, 56.5757027, -70.1978073, 56.5790672, -126.7543182, 126.7735138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5044876, upper bound: 84.3958918
time: 98.13 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5048751, upper bound: 84.4837350
time: 234.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -108.6746674, 78.7990341, -108.8380661, 78.8834381, -187.5581055, 187.6371002
1: -57.0590553, 59.0269699, -57.1578674, 59.0893021, -116.1483612, 116.1848373
2: -49.3538246, 60.4146690, -49.4227524, 60.4826965, -109.8365173, 109.8374176
3: -62.0134697, 73.3656464, -62.0938683, 73.4824371, -135.4959106, 135.4595032
4: -64.6841125, 70.6037598, -64.7723541, 70.7158890, -135.3999939, 135.3761139
5: -59.5923386, 73.0007553, -59.7017593, 73.1141586, -132.7064972, 132.7025146
6: -94.2096863, 62.8550835, -94.3411102, 62.9496460, -157.1593018, 157.1961975
7: -66.5094910, 69.4268265, -66.6488342, 69.5141449, -136.0236206, 136.0756531
8: -81.3423691, 83.7485657, -81.4230042, 83.8497162, -165.1920776, 165.1715698
9: -60.6733971, 76.6079941, -60.9641609, 76.7736816, -137.4470825, 137.5721588
10: -88.3617554, 90.9009933, -88.8567886, 91.1552582, -179.5170135, 179.7577820
11: -85.6967087, 58.0624466, -86.0232620, 58.1580582, -143.8547668, 144.0857086
12: -97.5385590, 77.0020370, -97.6741638, 77.0886993, -174.6272583, 174.6761932
13: -84.9780884, 98.4100037, -85.0863113, 98.5315323, -183.5096130, 183.4963074
14: -144.1422119, 82.0435486, -144.4826202, 82.2151718, -226.3573914, 226.5261688
15: -78.6221619, 64.2353668, -78.7059326, 64.3400421, -142.9622040, 142.9412994
16: -90.9251709, 72.2250671, -91.3100052, 72.3954468, -163.3206177, 163.5350647
17: -133.2620392, 71.6066132, -133.5871887, 71.7509689, -205.0130005, 205.1938019
18: -93.2546616, 69.7832947, -93.4002991, 69.8625946, -163.1172485, 163.1835785
19: -67.6452942, 40.3778458, -67.7707367, 40.4132957, -108.0585632, 108.1485825
20: -68.4953308, 53.0507584, -68.6070480, 53.1074409, -121.6027679, 121.6578064
21: -85.0122604, 51.1015930, -85.2163010, 51.1667175, -136.1789551, 136.3179016
22: -86.4569855, 46.3574104, -86.6259766, 46.4793205, -132.9363098, 132.9833679
23: -69.9520187, 54.0070496, -70.0818329, 54.0571060, -124.0091248, 124.0888748
24: -90.2524872, 54.3949356, -90.3658676, 54.4850006, -144.7374725, 144.7608032
25: -76.0794678, 55.4273911, -76.1738892, 55.5224648, -131.6019287, 131.6012573
26: -100.8447723, 82.0114441, -100.9924088, 82.0836487, -182.9284210, 183.0038452
27: -87.7367783, 49.5515251, -87.8718491, 49.6308289, -137.3676147, 137.4233704
28: -68.4466553, 54.4325180, -68.5203705, 54.4890404, -122.9356918, 122.9528809
29: -89.0526428, 41.7127190, -89.2117691, 41.7654190, -130.8180542, 130.9244843
30: -88.2528152, 63.5855560, -88.3916321, 63.6850357, -151.9378510, 151.9771729
31: -91.4674149, 56.0138245, -91.5974503, 56.0474205, -147.5148315, 147.6112671
32: -89.9117813, 57.4303169, -90.0307465, 57.5046310, -147.4163971, 147.4610596
33: -126.6004791, 77.8017426, -126.7795410, 78.1539917, -204.7544708, 204.5812836
34: -106.2955856, 48.5899429, -106.3968582, 48.7950974, -155.0906677, 154.9867859
35: -99.0592880, 58.6312103, -99.1854401, 58.9375839, -157.9968719, 157.8166504
36: -92.4400558, 57.0908813, -92.5802383, 57.3552856, -149.7953491, 149.6711121
37: -145.2454834, 62.5768356, -145.4985657, 62.8829231, -208.1284027, 208.0754089
38: -112.1831970, 71.0760040, -112.3679276, 71.4240112, -183.6072083, 183.4439392
39: -133.0318604, 76.3464813, -133.2611542, 76.6911469, -209.7229919, 209.6076355
40: -110.8679581, 56.7124939, -111.0512848, 56.9425316, -167.8104858, 167.7637787
41: -95.7417374, 65.7805481, -95.8897018, 65.9284744, -161.6701965, 161.6702423
42: -70.2192383, 56.5984535, -70.3413391, 56.6722794, -126.8915176, 126.9397888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5670240, upper bound: 84.3958918
time: 408.35 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5048751, upper bound: 84.4837350
time: 98.15 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -108.8654861, 78.9159241, -108.7742767, 78.8667755, -187.7322388, 187.6902008
1: -57.1713600, 59.1046333, -57.1132507, 59.0420647, -116.2134247, 116.2178802
2: -49.4076691, 60.4905014, -49.3431854, 60.4046669, -109.8123322, 109.8336868
3: -62.0719185, 73.5180054, -61.9814529, 73.3992081, -135.4711304, 135.4994507
4: -64.7850952, 70.7436218, -64.7222366, 70.6424179, -135.4275208, 135.4658508
5: -59.7074432, 73.1632309, -59.6322708, 73.0724716, -132.7799072, 132.7955017
6: -94.3371277, 62.9552040, -94.2095413, 62.8618622, -157.1989899, 157.1647491
7: -66.6510315, 69.5619354, -66.5553665, 69.4856262, -136.1366577, 136.1173096
8: -81.4262695, 83.8683548, -81.3570938, 83.7777557, -165.2040253, 165.2254486
9: -60.9706154, 76.8410950, -60.7521744, 76.7385712, -137.7091827, 137.5932617
10: -88.8647614, 91.2117538, -88.4042511, 91.0082245, -179.8729858, 179.6159973
11: -86.0418701, 58.1876945, -85.8267517, 58.1190224, -144.1608887, 144.0144348
12: -97.7056427, 77.0414047, -97.4382095, 76.9302750, -174.6359253, 174.4796143
13: -85.0915375, 98.5415573, -84.9738312, 98.4524536, -183.5439911, 183.5153809
14: -144.4908600, 82.2463531, -144.1393433, 82.0967407, -226.5876007, 226.3856964
15: -78.7051849, 64.3550339, -78.6149139, 64.2610168, -142.9662018, 142.9699402
16: -91.3254929, 72.4792328, -91.0943832, 72.3693085, -163.6947937, 163.5736084
17: -133.6106873, 71.7712402, -133.3935852, 71.6840515, -205.2947083, 205.1647949
18: -93.4052963, 69.8700714, -93.3015747, 69.7849503, -163.1902313, 163.1716309
19: -67.7866821, 40.4096680, -67.6743774, 40.3585434, -108.1452255, 108.0840454
20: -68.6183014, 53.1263542, -68.5332336, 53.0895691, -121.7078552, 121.6595917
21: -85.2382278, 51.1632843, -85.0706024, 51.0982361, -136.3364563, 136.2338867
22: -86.6994934, 46.4895706, -86.5969086, 46.4239349, -133.1234283, 133.0864868
23: -70.0993042, 54.0655823, -70.0067444, 54.0188484, -124.1181488, 124.0723267
24: -90.4066849, 54.4911270, -90.3469009, 54.4092751, -144.8159485, 144.8380280
25: -76.1933746, 55.5298615, -76.1165466, 55.4593849, -131.6527557, 131.6464081
26: -101.0053253, 82.0561981, -100.8480301, 81.9448853, -182.9501801, 182.9042358
27: -87.8693848, 49.6374168, -87.7628784, 49.5340118, -137.4033966, 137.4002991
28: -68.5063705, 54.4947510, -68.4274750, 54.3980179, -122.9043884, 122.9222260
29: -89.2431870, 41.7543793, -89.1443405, 41.7005539, -130.9437408, 130.8987122
30: -88.4072800, 63.7275810, -88.3267365, 63.6707916, -152.0780640, 152.0543060
31: -91.6088943, 56.0467796, -91.5019073, 55.9976158, -147.6065063, 147.5486908
32: -90.0589066, 57.5110016, -89.9671402, 57.4602966, -147.5191956, 147.4781494
33: -126.8615952, 78.1537323, -126.7510300, 77.8842468, -204.7458344, 204.9047546
34: -106.3758621, 48.8009300, -106.2597046, 48.5864716, -154.9623413, 155.0606232
35: -99.2059021, 58.9400024, -99.1051407, 58.6774139, -157.8833160, 158.0451355
36: -92.6149673, 57.3601913, -92.4960022, 57.1883354, -149.8032837, 149.8561707
37: -145.6657410, 62.8830299, -145.5492249, 62.7331390, -208.3988800, 208.4322510
38: -112.4124451, 71.4311066, -112.2679443, 71.1921692, -183.6045990, 183.6990509
39: -133.4010925, 76.6924210, -133.2844238, 76.5202637, -209.9213562, 209.9768066
40: -111.1237793, 56.9476166, -111.0103455, 56.8005486, -167.9243317, 167.9579620
41: -95.9431686, 65.9284515, -95.8387299, 65.8252487, -161.7684174, 161.7671661
42: -70.3735199, 56.6570282, -70.2746582, 56.5957565, -126.9692688, 126.9316864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5044876, upper bound: 84.4803429
time: 104.89 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5048751, upper bound: 84.5673684
time: 153.06 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -108.8934784, 78.9351349, -108.8932266, 78.9404831, -187.8339539, 187.8283691
1: -57.1847191, 59.1223946, -57.1803513, 59.1228638, -116.3075867, 116.3027496
2: -49.4546471, 60.5105057, -49.4555740, 60.5113792, -109.9660263, 109.9660797
3: -62.1350021, 73.5416718, -62.1305542, 73.5525589, -135.6875610, 135.6722260
4: -64.8189392, 70.7662888, -64.8221893, 70.7672577, -135.5861969, 135.5884705
5: -59.7418289, 73.1867218, -59.7348213, 73.1941605, -132.9359894, 132.9215393
6: -94.4330750, 62.9699631, -94.4349518, 62.9705925, -157.4036560, 157.4049072
7: -66.6869202, 69.5825348, -66.6791000, 69.5866318, -136.2735443, 136.2616272
8: -81.4539642, 83.8914795, -81.4523544, 83.8928986, -165.3468628, 165.3438110
9: -60.9945488, 76.9541016, -60.9953842, 76.9560318, -137.9505615, 137.9494781
10: -88.8958282, 91.4454117, -88.8968811, 91.4472046, -180.3430328, 180.3422852
11: -86.0672302, 58.2519760, -86.0675812, 58.2525368, -144.3197632, 144.3195496
12: -97.7377167, 77.1314545, -97.7394867, 77.1231995, -174.8609161, 174.8709412
13: -85.1272583, 98.5765228, -85.1287079, 98.5767136, -183.7039642, 183.7052307
14: -144.5325012, 82.4238358, -144.5336456, 82.4250793, -226.9575806, 226.9574738
15: -78.7361298, 64.3990326, -78.7364960, 64.3964233, -143.1325531, 143.1355133
16: -91.3628235, 72.5849838, -91.3637848, 72.5880280, -163.9508514, 163.9487610
17: -133.6385040, 71.8180542, -133.6400757, 71.8148193, -205.4533081, 205.4581146
18: -93.4361038, 69.9080734, -93.4477386, 69.9090958, -163.3451843, 163.3558044
19: -67.8184128, 40.4352760, -67.8202209, 40.4343338, -108.2527466, 108.2554932
20: -68.6444244, 53.1389694, -68.6446228, 53.1411934, -121.7856140, 121.7835922
21: -85.2725601, 51.2100067, -85.2734299, 51.2108002, -136.4833679, 136.4834290
22: -86.7469940, 46.5148315, -86.7481461, 46.5119514, -133.2589417, 133.2629700
23: -70.1235580, 54.0890694, -70.1246033, 54.0901146, -124.2136688, 124.2136612
24: -90.4519501, 54.5061035, -90.4626007, 54.5075226, -144.9594727, 144.9686890
25: -76.2186584, 55.5535889, -76.2194290, 55.5543175, -131.7729797, 131.7730103
26: -101.0375748, 82.1316223, -101.0490952, 82.1255341, -183.1631165, 183.1806946
27: -87.9621506, 49.6511612, -87.9721985, 49.6530380, -137.6151886, 137.6233521
28: -68.5593948, 54.5116081, -68.5591888, 54.5123711, -123.0717621, 123.0708008
29: -89.2858734, 41.7930183, -89.2888107, 41.7906647, -131.0765381, 131.0818176
30: -88.4342422, 63.7576981, -88.4342194, 63.7596474, -152.1938934, 152.1919250
31: -91.6465683, 56.0700493, -91.6527710, 56.0698128, -147.7163849, 147.7228088
32: -90.1162491, 57.5210075, -90.1201553, 57.5212631, -147.6374969, 147.6411591
33: -126.9840469, 78.1753387, -126.9871750, 78.1756134, -205.1596222, 205.1625061
34: -106.4951401, 48.8206024, -106.4978256, 48.8205566, -155.3157043, 155.3184204
35: -99.3196259, 58.9556770, -99.3229141, 58.9559669, -158.2755737, 158.2785950
36: -92.7251892, 57.3705597, -92.7273178, 57.3708038, -150.0959930, 150.0978699
37: -145.7450867, 62.9051743, -145.7521057, 62.9052277, -208.6502991, 208.6572723
38: -112.5445709, 71.4531860, -112.5472641, 71.4536667, -183.9982300, 184.0004425
39: -133.4790039, 76.7092819, -133.4850006, 76.7095642, -210.1885681, 210.1942749
40: -111.2139359, 56.9579735, -111.2169418, 56.9583511, -168.1722870, 168.1749115
41: -96.0264740, 65.9428711, -96.0291290, 65.9431610, -161.9696350, 161.9720001
42: -70.4178619, 56.6791878, -70.4188690, 56.6883621, -127.1062241, 127.0980530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5670240, upper bound: 84.4803428
time: 123.64 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5048751, upper bound: 84.5673684
time: 141.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 267.72 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 267.72
Output dim: 9, lower bound: -84.5044876, upper bound: 84.3958918
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 267.72
Output dim: 9, lower bound: -84.5048751, upper bound: 84.4837350
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 267.72
Output dim: 9, lower bound: -84.5670240, upper bound: 84.3958918
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 267.72
Output dim: 9, lower bound: -84.5048751, upper bound: 84.4837350
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 267.72
Output dim: 9, lower bound: -84.5044876, upper bound: 84.4803429
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 267.72
Output dim: 9, lower bound: -84.5048751, upper bound: 84.5673684
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 267.72
Output dim: 9, lower bound: -84.5670240, upper bound: 84.4803428
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 267.72
Output dim: 9, lower bound: -84.5048751, upper bound: 84.5673684

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -108.1864090, 78.5667877, -108.4802017, 78.7174301, -186.9038239, 187.0469666
1: -56.7010689, 58.8596191, -56.9154472, 58.9510727, -115.6521454, 115.7750626
2: -48.7547112, 60.2031441, -49.0233994, 60.3215408, -109.0762329, 109.2265472
3: -61.3576012, 73.0942688, -61.6257477, 73.2308273, -134.5884247, 134.7200165
4: -63.9393349, 70.3717804, -64.2869263, 70.5104599, -134.4497986, 134.6586914
5: -58.9993362, 72.6840363, -59.3008499, 72.8878708, -131.8872070, 131.9848938
6: -93.8568726, 62.5132599, -93.9764557, 62.6744995, -156.5313568, 156.4897156
7: -65.9361572, 69.2047577, -66.2530899, 69.3276062, -135.2637634, 135.4578552
8: -80.6718063, 83.4524841, -80.9950104, 83.6441040, -164.3159180, 164.4474945
9: -60.4072304, 75.8766479, -60.6336060, 76.2321625, -136.6393890, 136.5102539
10: -87.8500977, 89.4087982, -88.2309723, 90.0494537, -177.8995209, 177.6397705
11: -85.2711334, 57.1525726, -85.6638184, 57.5761757, -142.8473053, 142.8163757
12: -97.0668488, 75.6270142, -97.2584839, 76.2175140, -173.2843628, 172.8854980
13: -84.7328491, 97.8098984, -84.8381805, 98.1236877, -182.8565369, 182.6480713
14: -143.6184235, 80.7526093, -143.9267883, 81.2938843, -224.9123077, 224.6793976
15: -77.8745575, 63.9458542, -78.2160492, 64.0871277, -141.9616852, 142.1618958
16: -90.5149994, 71.4001007, -90.8851318, 71.7985611, -162.3135376, 162.2852173
17: -132.8604279, 70.6760025, -133.2318573, 71.1612854, -204.0217133, 203.9078522
18: -92.8307953, 69.4212646, -93.0751419, 69.5704346, -162.4012299, 162.4963989
19: -67.3577042, 40.2085419, -67.5133514, 40.2619095, -107.6196136, 107.7218781
20: -68.2201233, 52.7458992, -68.3923950, 52.8999290, -121.1200562, 121.1382904
21: -84.6737061, 50.7085266, -84.9019012, 50.8711700, -135.5448761, 135.6104126
22: -85.9538345, 46.1022072, -86.2478867, 46.2841759, -132.2380066, 132.3500977
23: -69.7307587, 53.7993240, -69.8753815, 53.8923645, -123.6231232, 123.6746979
24: -89.7188797, 54.2645416, -89.9983215, 54.3361549, -144.0550385, 144.2628479
25: -75.8291779, 55.1528320, -75.9594650, 55.3020706, -131.1312561, 131.1122894
26: -100.5058746, 81.2655258, -100.6608582, 81.5435028, -182.0493774, 181.9263916
27: -86.9669952, 49.3804741, -87.3205566, 49.4612503, -136.4282532, 136.7010193
28: -68.1136398, 54.2935257, -68.2510300, 54.3208237, -122.4344482, 122.5445480
29: -88.7336273, 41.3319778, -88.9387665, 41.4970360, -130.2306519, 130.2707367
30: -88.0053864, 63.1220093, -88.1776657, 63.3643265, -151.3697052, 151.2996674
31: -91.0296631, 55.7898827, -91.2578354, 55.8720360, -146.9017029, 147.0477142
32: -89.5680466, 56.9664040, -89.7562256, 57.2025223, -146.7705688, 146.7226257
33: -125.7800598, 77.4951935, -126.1765289, 77.7709579, -203.5509949, 203.6717224
34: -105.7780762, 48.3658485, -105.9509354, 48.4855309, -154.2635956, 154.3167725
35: -98.4247131, 58.3920975, -98.6938782, 58.5953979, -157.0201111, 157.0859680
36: -92.0121384, 56.9649315, -92.1886139, 57.1256752, -149.1377869, 149.1535492
37: -144.7085266, 62.3384552, -145.0683746, 62.6089935, -207.3175201, 207.4068298
38: -111.6777725, 70.8600159, -111.8957138, 71.0759583, -182.7537231, 182.7557373
39: -132.5239868, 76.1826630, -132.8464966, 76.4360657, -208.9600525, 209.0291595
40: -110.2962799, 56.5771408, -110.6088638, 56.7310448, -167.0272980, 167.1860046
41: -95.3575592, 65.6319122, -95.5523300, 65.7436066, -161.1011658, 161.1842346
42: -69.9376068, 56.0739746, -70.0891571, 56.3160400, -126.2536316, 126.1631317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1036

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4989034, upper bound: 84.3516173
time: 311.40 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4989034, upper bound: 84.3903357
time: 125.42 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -108.6345673, 78.7749939, -108.7159348, 78.8075714, -187.4421387, 187.4909363
1: -57.0339088, 59.0053864, -57.0871849, 59.0071106, -116.0410156, 116.0925751
2: -49.2951241, 60.3906479, -49.3076782, 60.3744736, -109.6696014, 109.6983261
3: -61.9400139, 73.3363800, -61.9421806, 73.3255997, -135.2656097, 135.2785645
4: -64.6352081, 70.5748901, -64.6683807, 70.5883484, -135.2235565, 135.2432709
5: -59.5470810, 72.9723969, -59.5970955, 72.9851379, -132.5322266, 132.5694885
6: -94.1060028, 62.8097153, -94.1139908, 62.8298759, -156.9358521, 156.9237061
7: -66.4597702, 69.4001770, -66.5221710, 69.4057617, -135.8655243, 135.9223480
8: -81.2989044, 83.7195892, -81.3230972, 83.7321930, -165.0310974, 165.0426788
9: -60.6441689, 76.4791336, -60.7188148, 76.5506058, -137.1947784, 137.1979523
10: -88.3236694, 90.6362305, -88.3616791, 90.7054672, -179.0290985, 178.9979095
11: -85.6638336, 57.9783325, -85.7794266, 58.0181198, -143.6819458, 143.7577515
12: -97.4992065, 76.8893814, -97.3691406, 76.8886032, -174.3878174, 174.2585144
13: -84.9327927, 98.3618927, -84.9277191, 98.4027100, -183.3355103, 183.2895966
14: -144.0889587, 81.8595123, -144.0839844, 81.8911362, -225.9801025, 225.9434814
15: -78.5734482, 64.1858521, -78.5779953, 64.2028427, -142.7762756, 142.7638397
16: -90.8777390, 72.0975037, -91.0366135, 72.1691437, -163.0468750, 163.1341248
17: -133.2240601, 71.5467377, -133.3366699, 71.6149445, -204.8390045, 204.8834076
18: -93.2006760, 69.7272110, -93.2459106, 69.7319489, -162.9326172, 162.9731140
19: -67.6038132, 40.3463936, -67.6214752, 40.3360367, -107.9398422, 107.9678497
20: -68.4641647, 53.0294914, -68.4938889, 53.0524750, -121.5166321, 121.5233765
21: -84.9709778, 51.0485916, -85.0110016, 51.0527115, -136.0236816, 136.0596008
22: -86.3818665, 46.3238449, -86.4650726, 46.3890610, -132.7709351, 132.7889099
23: -69.9226227, 53.9688034, -69.9620590, 53.9809036, -123.9035187, 123.9308472
24: -90.1996918, 54.3752365, -90.2528229, 54.3849754, -144.5846710, 144.6280518
25: -76.0430756, 55.3938599, -76.0669861, 55.4240570, -131.4671173, 131.4608459
26: -100.8031845, 81.9161072, -100.7857895, 81.8964691, -182.6996460, 182.7019043
27: -87.6353760, 49.5322609, -87.6657944, 49.5097313, -137.1451111, 137.1980591
28: -68.3880005, 54.4110374, -68.3873978, 54.3729935, -122.7609787, 122.7984314
29: -88.9993362, 41.6667862, -89.0639801, 41.6733475, -130.6726685, 130.7307739
30: -88.2172318, 63.5410881, -88.2814178, 63.5906944, -151.8079224, 151.8224792
31: -91.4159622, 55.9822197, -91.4416199, 55.9726524, -147.3886108, 147.4238434
32: -89.8498840, 57.4069748, -89.8775482, 57.4384918, -147.2883606, 147.2845154
33: -126.4608307, 77.7754059, -126.5368881, 77.8607025, -204.3215179, 204.3122864
34: -106.1659012, 48.5655327, -106.1547089, 48.5591888, -154.7250977, 154.7202454
35: -98.9364777, 58.6128578, -98.9677582, 58.6579514, -157.5944214, 157.5806122
36: -92.3174744, 57.0776215, -92.3449554, 57.1717339, -149.4892120, 149.4225769
37: -145.1485291, 62.5478363, -145.2893982, 62.7082939, -207.8568268, 207.8372192
38: -112.0348587, 71.0476532, -112.0831757, 71.1600037, -183.1948547, 183.1308289
39: -132.9326477, 76.3244858, -133.0532837, 76.4998779, -209.4325256, 209.3777771
40: -110.7677765, 56.6939201, -110.8413086, 56.7815933, -167.5493774, 167.5352325
41: -95.6527405, 65.7596436, -95.6978607, 65.8080444, -161.4607849, 161.4575043
42: -70.1679611, 56.5422440, -70.1948471, 56.5654945, -126.7334595, 126.7370911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1036

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4993118, upper bound: 84.4393723
time: 117.38 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4993118, upper bound: 84.4781836
time: 131.15 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -108.2161484, 78.5858612, -108.5986328, 78.7912064, -187.0073547, 187.1844940
1: -56.7163582, 58.8777428, -56.9826241, 59.0317688, -115.7481232, 115.8603516
2: -48.8003502, 60.2230911, -49.1336288, 60.4283867, -109.2287369, 109.3567200
3: -61.4222679, 73.1175232, -61.7742195, 73.3856964, -134.8079681, 134.8917389
4: -63.9703369, 70.3945541, -64.3846283, 70.6357727, -134.6060944, 134.7791748
5: -59.0369568, 72.7069550, -59.4030457, 73.0150757, -132.0520325, 132.1100006
6: -93.9523926, 62.5280914, -94.1998444, 62.7831688, -156.7355652, 156.7279358
7: -65.9762650, 69.2248840, -66.3760376, 69.4337997, -135.4100647, 135.6009216
8: -80.6996613, 83.4760895, -81.0888977, 83.7597809, -164.4594421, 164.5649872
9: -60.4314270, 75.9904022, -60.8772545, 76.4497986, -136.8812103, 136.8676605
10: -87.8820648, 89.6437225, -88.7241211, 90.4886246, -178.3706665, 178.3678436
11: -85.2969666, 57.2158585, -85.9050980, 57.7084045, -143.0053711, 143.1209564
12: -97.0989532, 75.7196655, -97.5611649, 76.4101791, -173.5091248, 173.2808228
13: -84.7694855, 97.8444824, -84.9936829, 98.2476044, -183.0170898, 182.8381653
14: -143.6609802, 80.9187012, -144.3217468, 81.6113892, -225.2723541, 225.2404480
15: -77.9055939, 63.9896240, -78.3377838, 64.2215271, -142.1271057, 142.3274078
16: -90.5532074, 71.5067062, -91.1554031, 72.0172424, -162.5704498, 162.6621094
17: -132.8891602, 70.7223969, -133.4790955, 71.2913513, -204.1805115, 204.2014771
18: -92.8621292, 69.4596710, -93.2211685, 69.6945648, -162.5567017, 162.6808319
19: -67.3893585, 40.2349281, -67.6590881, 40.3373566, -107.7266998, 107.8940125
20: -68.2466049, 52.7585602, -68.5034866, 52.9517059, -121.1983032, 121.2620316
21: -84.7083969, 50.7555923, -85.1047821, 50.9830170, -135.6914062, 135.8603821
22: -85.9998703, 46.1287956, -86.3981094, 46.3718376, -132.3717041, 132.5269012
23: -69.7553940, 53.8220901, -69.9935150, 53.9626427, -123.7180176, 123.8156052
24: -89.7601395, 54.2793312, -90.1069412, 54.4343185, -144.1944580, 144.3862610
25: -75.8545074, 55.1768723, -76.0624542, 55.3972740, -131.2517700, 131.2393188
26: -100.5384903, 81.3425217, -100.8644867, 81.7240601, -182.2625427, 182.2070007
27: -87.0526428, 49.3944168, -87.5205536, 49.5804176, -136.6330566, 136.9149780
28: -68.1662064, 54.3099480, -68.3817139, 54.4349098, -122.6011200, 122.6916656
29: -88.7755203, 41.3716660, -89.0822754, 41.5867233, -130.3622437, 130.4539490
30: -88.0341187, 63.1507530, -88.2850647, 63.4533081, -151.4874268, 151.4358063
31: -91.0669861, 55.8135414, -91.4088669, 55.9435387, -147.0105286, 147.2224121
32: -89.6236038, 56.9771767, -89.9064407, 57.2639618, -146.8875732, 146.8836060
33: -125.9010849, 77.5168610, -126.4122620, 78.0625610, -203.9636536, 203.9291229
34: -105.8974075, 48.3860779, -106.1890869, 48.7200928, -154.6174622, 154.5751648
35: -98.5324402, 58.4081726, -98.9060211, 58.8742752, -157.4067078, 157.3141785
36: -92.1215668, 56.9755630, -92.4189377, 57.3083572, -149.4299164, 149.3945007
37: -144.7869873, 62.3606262, -145.2705383, 62.7811813, -207.5681763, 207.6311646
38: -111.8122253, 70.8829803, -112.1753693, 71.3382874, -183.1504974, 183.0583496
39: -132.6007996, 76.1997299, -133.0458221, 76.6254807, -209.2262268, 209.2455444
40: -110.3846512, 56.5877380, -110.8141479, 56.8889580, -167.2736053, 167.4018860
41: -95.4373474, 65.6469803, -95.7403717, 65.8618546, -161.2991943, 161.3873444
42: -69.9819031, 56.0964546, -70.2325134, 56.4096489, -126.3915329, 126.3289642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1036

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4989034, upper bound: 84.3516173
time: 124.88 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4989034, upper bound: 84.3903357
time: 107.90 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -108.6641312, 78.7939529, -108.8340378, 78.8814163, -187.5455322, 187.6279907
1: -57.0491486, 59.0231247, -57.1542015, 59.0878563, -116.1370087, 116.1773224
2: -49.3406448, 60.4104958, -49.4179192, 60.4811134, -109.8217545, 109.8284149
3: -62.0037994, 73.3594208, -62.0901375, 73.4800568, -135.4838562, 135.4495544
4: -64.6663513, 70.5973206, -64.7658768, 70.7133484, -135.3796997, 135.3631897
5: -59.5835304, 72.9948730, -59.6983604, 73.1119080, -132.6954193, 132.6932373
6: -94.2013321, 62.8246880, -94.3377075, 62.9385719, -157.1399078, 157.1623840
7: -66.5001373, 69.4198608, -66.6452942, 69.5114899, -136.0116272, 136.0651398
8: -81.3267441, 83.7429276, -81.4171448, 83.8475189, -165.1742554, 165.1600647
9: -60.6680756, 76.5925446, -60.9621201, 76.7679825, -137.4360657, 137.5546570
10: -88.3551102, 90.8707733, -88.8542175, 91.1441803, -179.4992981, 179.7249908
11: -85.6895447, 58.0415001, -86.0204620, 58.1502762, -143.8398132, 144.0619659
12: -97.5309906, 76.9818420, -97.6711426, 77.0813751, -174.6123657, 174.6529694
13: -84.9695206, 98.3965759, -85.0830383, 98.5265045, -183.4960327, 183.4795990
14: -144.1312408, 82.0256195, -144.4784088, 82.2086182, -226.3398438, 226.5040283
15: -78.6044312, 64.2295837, -78.6993256, 64.3378296, -142.9422607, 142.9289093
16: -90.9154358, 72.2038879, -91.3062439, 72.3875961, -163.3030396, 163.5101318
17: -133.2526245, 71.5933533, -133.5836182, 71.7449875, -204.9976196, 205.1769562
18: -93.2320557, 69.7654114, -93.3919907, 69.8560181, -163.0880737, 163.1574097
19: -67.6351624, 40.3722343, -67.7670441, 40.4108315, -108.0459900, 108.1392746
20: -68.4903717, 53.0421219, -68.6051025, 53.1040878, -121.5944519, 121.6472244
21: -85.0053787, 51.0952034, -85.2136993, 51.1642914, -136.1696777, 136.3088989
22: -86.4283524, 46.3504791, -86.6153793, 46.4766235, -132.9049683, 132.9658508
23: -69.9470139, 53.9917145, -70.0799255, 54.0512314, -123.9982376, 124.0716324
24: -90.2410507, 54.3901672, -90.3616028, 54.4830971, -144.7241516, 144.7517700
25: -76.0685883, 55.4178200, -76.1698761, 55.5188751, -131.5874634, 131.5876923
26: -100.8355789, 81.9925537, -100.9888535, 82.0767136, -182.9122925, 182.9814148
27: -87.7213211, 49.5461349, -87.8660431, 49.6288223, -137.3501434, 137.4121704
28: -68.4407959, 54.4279251, -68.5182190, 54.4872742, -122.9280701, 122.9461441
29: -89.0410919, 41.7065773, -89.2073822, 41.7629852, -130.8040771, 130.9139404
30: -88.2455521, 63.5708275, -88.3888168, 63.6795731, -151.9251251, 151.9596405
31: -91.4532471, 56.0054359, -91.5922318, 56.0440445, -147.4972839, 147.5976715
32: -89.9051971, 57.4178123, -90.0280609, 57.4998627, -147.4050598, 147.4458771
33: -126.5819016, 77.7968750, -126.7726593, 78.1520081, -204.7339172, 204.5695190
34: -106.2852783, 48.5854836, -106.3929672, 48.7934036, -155.0786743, 154.9784546
35: -99.0441437, 58.6287231, -99.1798706, 58.9365540, -157.9806976, 157.8085938
36: -92.4269333, 57.0880737, -92.5753326, 57.3542252, -149.7811432, 149.6634064
37: -145.2272034, 62.5699463, -145.4918060, 62.8803482, -208.1075439, 208.0617523
38: -112.1687012, 71.0703888, -112.3624268, 71.4219589, -183.5906677, 183.4328003
39: -133.0096436, 76.3414688, -133.2529907, 76.6891785, -209.6987915, 209.5944519
40: -110.8565216, 56.7045059, -111.0466690, 56.9395828, -167.7960815, 167.7511749
41: -95.7325058, 65.7745667, -95.8861313, 65.9262543, -161.6587524, 161.6606750
42: -70.2119522, 56.5650063, -70.3383636, 56.6587563, -126.8706970, 126.9033661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1036

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4993118, upper bound: 84.4393723
time: 106.27 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4993118, upper bound: 84.4781837
time: 111.06 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -108.4074249, 78.7023163, -108.5344696, 78.7745361, -187.1819611, 187.2367706
1: -56.8289452, 58.9558640, -56.9377480, 58.9848557, -115.8137894, 115.8936157
2: -48.8538322, 60.2995262, -49.0538940, 60.3504410, -109.2042694, 109.3534241
3: -61.4806404, 73.2689285, -61.6611176, 73.3021088, -134.7827454, 134.9300537
4: -64.0714264, 70.5365143, -64.3341675, 70.5623474, -134.6337738, 134.8706818
5: -59.1518288, 72.8680878, -59.3322334, 72.9730682, -132.1248932, 132.2003174
6: -94.0793762, 62.6285629, -94.0689163, 62.6952896, -156.7746582, 156.6974792
7: -66.1183701, 69.3598938, -66.2826996, 69.4050293, -135.5234070, 135.6425781
8: -80.7837677, 83.5966644, -81.0230713, 83.6879272, -164.4716949, 164.6197357
9: -60.7288971, 76.2225037, -60.6649933, 76.4142075, -137.1430969, 136.8874969
10: -88.3859711, 89.9535522, -88.2713394, 90.3409576, -178.7268982, 178.2248840
11: -85.6426392, 57.3406944, -85.7084579, 57.6691017, -143.3117371, 143.0491333
12: -97.2665100, 75.7595825, -97.3250351, 76.2520523, -173.5185394, 173.0846100
13: -84.8832169, 97.9765778, -84.8811417, 98.1684952, -183.0517120, 182.8577271
14: -144.0096588, 81.1210480, -143.9779205, 81.4928894, -225.5025482, 225.0989685
15: -77.9891510, 64.1096802, -78.2462921, 64.1432037, -142.1323547, 142.3559723
16: -90.9539795, 71.7602463, -90.9394150, 71.9907303, -162.9447021, 162.6996613
17: -133.2377930, 70.8880157, -133.2853241, 71.2244492, -204.4622498, 204.1733398
18: -93.0129852, 69.5463257, -93.1224899, 69.6167984, -162.6297760, 162.6688232
19: -67.5306549, 40.2662544, -67.5628357, 40.2822189, -107.8128738, 107.8290863
20: -68.3694916, 52.8340530, -68.4298859, 52.9336395, -121.3031158, 121.2639389
21: -84.9344177, 50.8168182, -84.9589844, 50.9142990, -135.8487244, 135.7757874
22: -86.2420120, 46.2616768, -86.3688507, 46.3163338, -132.5583496, 132.6305237
23: -69.9026642, 53.8804054, -69.9182892, 53.9242668, -123.8269348, 123.7986908
24: -89.9139938, 54.3758774, -90.0879364, 54.3585587, -144.2725525, 144.4638062
25: -75.9689255, 55.2798271, -76.0049210, 55.3337631, -131.3026886, 131.2847443
26: -100.6987457, 81.3871155, -100.7197037, 81.5849686, -182.2837067, 182.1068115
27: -87.1849365, 49.4805374, -87.4114380, 49.4836655, -136.6685944, 136.8919678
28: -68.2261963, 54.3731384, -68.2889709, 54.3442345, -122.5704117, 122.6621094
29: -88.9662018, 41.4137306, -89.0148773, 41.5217171, -130.4879150, 130.4286041
30: -88.1884384, 63.2927246, -88.2202606, 63.4386139, -151.6270447, 151.5129700
31: -91.2087708, 55.8461685, -91.3131638, 55.8936119, -147.1023712, 147.1593323
32: -89.7703705, 57.0581055, -89.8432083, 57.2195282, -146.9898834, 146.9013062
33: -126.1617889, 77.8688507, -126.3836670, 77.7925644, -203.9543457, 204.2525177
34: -105.9774628, 48.5970917, -106.0519333, 48.5112991, -154.4887695, 154.6490173
35: -98.6786346, 58.7172203, -98.8255463, 58.6140099, -157.2926331, 157.5427551
36: -92.2959595, 57.2448997, -92.3345871, 57.1414185, -149.4373627, 149.5794830
37: -145.2068176, 62.6669998, -145.3211060, 62.6311722, -207.8379669, 207.9880981
38: -112.0398560, 71.2383270, -112.0746231, 71.1063080, -183.1461487, 183.3129578
39: -132.9692993, 76.5456161, -133.0690613, 76.4545975, -209.4238892, 209.6146851
40: -110.6401291, 56.8229980, -110.7729568, 56.7470398, -167.3871765, 167.5959473
41: -95.6382523, 65.7948456, -95.6895828, 65.7585983, -161.3968506, 161.4844208
42: -70.1370697, 56.1557312, -70.1666260, 56.3326607, -126.4697266, 126.3223572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1036

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4989034, upper bound: 84.4360997
time: 114.57 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4989034, upper bound: 84.4747943
time: 259.69 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -108.8549652, 78.9108047, -108.7703094, 78.8647614, -187.7197113, 187.6811218
1: -57.1615143, 59.1008339, -57.1095619, 59.0406227, -116.2021332, 116.2103958
2: -49.3945122, 60.4863815, -49.3383446, 60.4030876, -109.7975998, 109.8247223
3: -62.0621719, 73.5117111, -61.9776917, 73.3968048, -135.4589844, 135.4894104
4: -64.7673340, 70.7372284, -64.7157440, 70.6398849, -135.4072266, 135.4529724
5: -59.6986923, 73.1573181, -59.6288986, 73.0701981, -132.7688904, 132.7861938
6: -94.3287277, 62.9248238, -94.2061462, 62.8507614, -157.1794891, 157.1309662
7: -66.6416931, 69.5549393, -66.5518341, 69.4829712, -136.1246643, 136.1067657
8: -81.4106598, 83.8627319, -81.3512878, 83.7755737, -165.1862335, 165.2140198
9: -60.9652901, 76.8256378, -60.7501068, 76.7328644, -137.6981506, 137.5757446
10: -88.8581085, 91.1814423, -88.4016418, 90.9971237, -179.8552246, 179.5830688
11: -86.0347214, 58.1667328, -85.8239288, 58.1112137, -144.1459351, 143.9906616
12: -97.6980591, 77.0213089, -97.4351730, 76.9228973, -174.6209564, 174.4564667
13: -85.0829773, 98.5281143, -84.9706497, 98.4474182, -183.5303955, 183.4987640
14: -144.4798431, 82.2284317, -144.1350708, 82.0902100, -226.5700378, 226.3634949
15: -78.6875153, 64.3492508, -78.6082916, 64.2587738, -142.9462738, 142.9575500
16: -91.3158112, 72.4580383, -91.0906448, 72.3614731, -163.6772766, 163.5486755
17: -133.6012268, 71.7579803, -133.3900146, 71.6780701, -205.2792969, 205.1479950
18: -93.3827515, 69.8522186, -93.2931976, 69.7783051, -163.1610565, 163.1454163
19: -67.7766571, 40.4042435, -67.6706696, 40.3563995, -108.1330566, 108.0749130
20: -68.6133575, 53.1177444, -68.5313110, 53.0862427, -121.6996002, 121.6490402
21: -85.2313309, 51.1568794, -85.0679398, 51.0958405, -136.3271790, 136.2248230
22: -86.6708679, 46.4826393, -86.5862885, 46.4211731, -133.0920410, 133.0689240
23: -70.0942917, 54.0501556, -70.0048370, 54.0129700, -124.1072540, 124.0549927
24: -90.3952255, 54.4863892, -90.3426361, 54.4073792, -144.8026123, 144.8290100
25: -76.1824951, 55.5203552, -76.1125031, 55.4557991, -131.6382751, 131.6328583
26: -100.9961090, 82.0374222, -100.8445053, 81.9379654, -182.9340820, 182.8819275
27: -87.8539429, 49.6320457, -87.7570572, 49.5320129, -137.3859558, 137.3890991
28: -68.5005035, 54.4902153, -68.4253159, 54.3962975, -122.8968048, 122.9155273
29: -89.2316895, 41.7482529, -89.1399078, 41.6980896, -130.9297638, 130.8881531
30: -88.3999786, 63.7128754, -88.3239059, 63.6653366, -152.0653076, 152.0367737
31: -91.5947495, 56.0385017, -91.4966812, 55.9942131, -147.5889587, 147.5351868
32: -90.0523300, 57.4985046, -89.9644775, 57.4555321, -147.5078430, 147.4629822
33: -126.8429947, 78.1487808, -126.7441788, 77.8822784, -204.7252808, 204.8929443
34: -106.3655624, 48.7964554, -106.2558060, 48.5848083, -154.9503784, 155.0522614
35: -99.1906967, 58.9375572, -99.0996017, 58.6763420, -157.8670349, 158.0371552
36: -92.6018295, 57.3573685, -92.4910812, 57.1873016, -149.7891235, 149.8484497
37: -145.6474915, 62.8761864, -145.5424194, 62.7304878, -208.3779755, 208.4186096
38: -112.3979340, 71.4255753, -112.2623291, 71.1901169, -183.5880432, 183.6878967
39: -133.3788757, 76.6874084, -133.2762451, 76.5183105, -209.8971863, 209.9636383
40: -111.1122360, 56.9396172, -111.0059052, 56.7975693, -167.9098053, 167.9455261
41: -95.9339294, 65.9225006, -95.8351364, 65.8229828, -161.7568970, 161.7576294
42: -70.3662720, 56.6235352, -70.2716980, 56.5822372, -126.9485016, 126.8952332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1036

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4993118, upper bound: 84.5230785
time: 184.43 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4993118, upper bound: 84.5618419
time: 99.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -108.4354935, 78.7217255, -108.6536102, 78.8481750, -187.2836609, 187.3753357
1: -56.8423653, 58.9740143, -57.0050735, 59.0656509, -115.9079971, 115.9790878
2: -48.9007492, 60.3197136, -49.1662674, 60.4573402, -109.3580933, 109.4859772
3: -61.5446243, 73.2930069, -61.8107529, 73.4556122, -135.0002136, 135.1037598
4: -64.1048584, 70.5595398, -64.4341202, 70.6875381, -134.7923889, 134.9936523
5: -59.1867981, 72.8920975, -59.4357681, 73.0952911, -132.2820740, 132.3278656
6: -94.1755829, 62.6431313, -94.2939911, 62.8039474, -156.9794922, 156.9371185
7: -66.1543655, 69.3810425, -66.4061661, 69.5061493, -135.6605225, 135.7872009
8: -80.8113632, 83.6200790, -81.1182251, 83.8034210, -164.6147766, 164.7383118
9: -60.7532234, 76.3358688, -60.9086151, 76.6319122, -137.3851318, 137.2444763
10: -88.4176025, 90.1876526, -88.7646027, 90.7804108, -179.1979980, 178.9522400
11: -85.6681824, 57.4050140, -85.9496155, 57.8026581, -143.4708252, 143.3546295
12: -97.2989502, 75.8498077, -97.6270294, 76.4449158, -173.7438660, 173.4768372
13: -84.9188538, 98.0114517, -85.0360489, 98.2928314, -183.2116699, 183.0475006
14: -144.0516663, 81.2986679, -144.3727112, 81.8211746, -225.8728333, 225.6713867
15: -78.0200195, 64.1537704, -78.3682556, 64.2781754, -142.2981567, 142.5220337
16: -90.9918671, 71.8661575, -91.2094421, 72.2096863, -163.2015381, 163.0755920
17: -133.2657928, 70.9345856, -133.5321045, 71.3552856, -204.6210785, 204.4666901
18: -93.0437088, 69.5844116, -93.2686462, 69.7410660, -162.7847748, 162.8530579
19: -67.5625610, 40.2921448, -67.7088242, 40.3583450, -107.9209061, 108.0009689
20: -68.3958588, 52.8466682, -68.5411530, 52.9854012, -121.3812561, 121.3878098
21: -84.9691086, 50.8639374, -85.1620865, 51.0270576, -135.9961548, 136.0260315
22: -86.2892990, 46.2869568, -86.5199661, 46.4044952, -132.6937866, 132.8069153
23: -69.9271927, 53.9038429, -70.0364075, 53.9955292, -123.9227142, 123.9402390
24: -89.9591751, 54.3907509, -90.2034683, 54.4568596, -144.4160309, 144.5942078
25: -75.9940491, 55.3036308, -76.1079330, 55.4291000, -131.4231415, 131.4115601
26: -100.7312851, 81.4628830, -100.9214249, 81.7659149, -182.4971924, 182.3843079
27: -87.2775040, 49.4943771, -87.6205826, 49.6027031, -136.8802032, 137.1149597
28: -68.2790222, 54.3896065, -68.4205246, 54.4583740, -122.7373886, 122.8101273
29: -89.0089722, 41.4522324, -89.1594620, 41.6118469, -130.6208191, 130.6116943
30: -88.2159805, 63.3219261, -88.3277740, 63.5276184, -151.7435913, 151.6497040
31: -91.2464218, 55.8696671, -91.4644165, 55.9660034, -147.2124329, 147.3340759
32: -89.8279266, 57.0680656, -89.9958801, 57.2805176, -147.1084442, 147.0639343
33: -126.2841263, 77.8907394, -126.6196518, 78.0842361, -204.3683624, 204.5103912
34: -106.0967484, 48.6171570, -106.2899094, 48.7456665, -154.8424072, 154.9070587
35: -98.7924042, 58.7331543, -99.0433197, 58.8928375, -157.6852417, 157.7764587
36: -92.4061508, 57.2554321, -92.5658417, 57.3241119, -149.7302551, 149.8212738
37: -145.2859192, 62.6890793, -145.5237427, 62.8034439, -208.0893555, 208.2128296
38: -112.1725769, 71.2606430, -112.3544540, 71.3682251, -183.5408020, 183.6150970
39: -133.0468750, 76.5626068, -133.2692261, 76.6440125, -209.6908875, 209.8318329
40: -110.7300034, 56.8333664, -110.9794006, 56.9048462, -167.6348419, 167.8127747
41: -95.7215347, 65.8093185, -95.8797150, 65.8765335, -161.5980682, 161.6890259
42: -70.1819382, 56.1776237, -70.3106995, 56.4256058, -126.6075439, 126.4883194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1036

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4989034, upper bound: 84.4360997
time: 117.84 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4989034, upper bound: 84.4747943
time: 121.68 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -108.8829193, 78.9300613, -108.8892136, 78.9384766, -187.8213959, 187.8192596
1: -57.1748581, 59.1185684, -57.1766739, 59.1214256, -116.2962799, 116.2952423
2: -49.4414368, 60.5063934, -49.4507370, 60.5098152, -109.9512482, 109.9571228
3: -62.1253548, 73.5354004, -62.1268044, 73.5501709, -135.6755219, 135.6622009
4: -64.8011627, 70.7598953, -64.8156738, 70.7647324, -135.5658875, 135.5755615
5: -59.7330894, 73.1808090, -59.7314758, 73.1918869, -132.9249725, 132.9122620
6: -94.4247284, 62.9396019, -94.4316101, 62.9594994, -157.3842316, 157.3712006
7: -66.6775818, 69.5755692, -66.6755829, 69.5839996, -136.2615814, 136.2511444
8: -81.4383469, 83.8858337, -81.4465103, 83.8907318, -165.3290710, 165.3323364
9: -60.9892235, 76.9386520, -60.9933281, 76.9503174, -137.9395447, 137.9319763
10: -88.8892059, 91.4150925, -88.8942719, 91.4361420, -180.3253174, 180.3093567
11: -86.0600510, 58.2310333, -86.0647507, 58.2447510, -144.3047943, 144.2957764
12: -97.7301788, 77.1112900, -97.7364807, 77.1158905, -174.8460693, 174.8477631
13: -85.1187286, 98.5630722, -85.1254578, 98.5716553, -183.6903839, 183.6885071
14: -144.5215454, 82.4059448, -144.5294037, 82.4185715, -226.9401245, 226.9353485
15: -78.7184525, 64.3932571, -78.7298737, 64.3941803, -143.1126251, 143.1231384
16: -91.3531876, 72.5637665, -91.3600693, 72.5801544, -163.9333496, 163.9238281
17: -133.6291199, 71.8047867, -133.6365662, 71.8088455, -205.4379578, 205.4413300
18: -93.4135132, 69.8901520, -93.4394073, 69.9024353, -163.3159485, 163.3295593
19: -67.8082809, 40.4296646, -67.8165131, 40.4318886, -108.2401581, 108.2461700
20: -68.6394882, 53.1303635, -68.6426849, 53.1378593, -121.7773438, 121.7730408
21: -85.2657242, 51.2036057, -85.2707748, 51.2084045, -136.4741211, 136.4743652
22: -86.7183609, 46.5079193, -86.7375031, 46.5092354, -133.2275696, 133.2454224
23: -70.1185608, 54.0737610, -70.1227036, 54.0842590, -124.2028198, 124.1964645
24: -90.4404678, 54.5013428, -90.4582901, 54.5056381, -144.9461060, 144.9596252
25: -76.2077789, 55.5440445, -76.2154236, 55.5507660, -131.7585449, 131.7594604
26: -101.0283813, 82.1127472, -101.0455704, 82.1185760, -183.1469421, 183.1583252
27: -87.9466629, 49.6457977, -87.9663849, 49.6510468, -137.5977020, 137.6121826
28: -68.5535431, 54.5070534, -68.5570374, 54.5105972, -123.0641403, 123.0640869
29: -89.2743607, 41.7868881, -89.2844086, 41.7882004, -131.0625610, 131.0712891
30: -88.4269867, 63.7429352, -88.4313965, 63.7541580, -152.1811371, 152.1743317
31: -91.6324081, 56.0616531, -91.6475830, 56.0664444, -147.6988525, 147.7092285
32: -90.1096420, 57.5084991, -90.1174698, 57.5164986, -147.6261444, 147.6259613
33: -126.9654846, 78.1704407, -126.9803085, 78.1736603, -205.1391144, 205.1507568
34: -106.4848480, 48.8161736, -106.4939270, 48.8188629, -155.3037109, 155.3100891
35: -99.3044434, 58.9532585, -99.3173447, 58.9549561, -158.2593994, 158.2705994
36: -92.7120667, 57.3677826, -92.7224121, 57.3697395, -150.0818024, 150.0901794
37: -145.7268372, 62.8982544, -145.7453308, 62.9026031, -208.6294403, 208.6435852
38: -112.5300827, 71.4476242, -112.5417328, 71.4515839, -183.9816589, 183.9893494
39: -133.4566650, 76.7042847, -133.4768066, 76.7076416, -210.1643066, 210.1810913
40: -111.2024307, 56.9500046, -111.2124176, 56.9553108, -168.1577301, 168.1624146
41: -96.0172119, 65.9368896, -96.0255280, 65.9409180, -161.9581146, 161.9624023
42: -70.4106598, 56.6457596, -70.4159088, 56.6748543, -127.0855103, 127.0616608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 919
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 920
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 746
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1470
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 922
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1029
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1028
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 921
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1626
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1034
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1528
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1038
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1030
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1035
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1032
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1039
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1232
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1535
type: B, layer: 1, pos: 791
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1248
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1264
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1036

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1757

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4993118, upper bound: 84.5230785
time: 244.01 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4993118, upper bound: 84.5618419
time: 127.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 373.43 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4989034, upper bound: 84.3516173
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4989034, upper bound: 84.3903357
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4993118, upper bound: 84.4393723
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4993118, upper bound: 84.4781836
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4989034, upper bound: 84.3516173
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4989034, upper bound: 84.3903357
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4993118, upper bound: 84.4393723
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4993118, upper bound: 84.4781837
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4989034, upper bound: 84.4360997
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4989034, upper bound: 84.4747943
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4993118, upper bound: 84.5230785
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4993118, upper bound: 84.5618419
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4989034, upper bound: 84.4360997
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4989034, upper bound: 84.4747943
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4993118, upper bound: 84.5230785
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 373.43
Output dim: 9, lower bound: -84.4993118, upper bound: 84.5618419

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -107.9500275, 78.5344620, -108.0910873, 78.6640320, -186.6140594, 186.6255493
1: -56.5357361, 58.8425598, -56.6410675, 58.9229584, -115.4586945, 115.4836273
2: -48.6416817, 60.1850662, -48.8358917, 60.2917175, -108.9333954, 109.0209503
3: -61.2372513, 73.0704956, -61.4259758, 73.1917267, -134.4289856, 134.4964600
4: -63.7824593, 70.3521881, -64.0270920, 70.4781647, -134.2606201, 134.3792725
5: -58.8647652, 72.6541595, -59.0774689, 72.8385162, -131.7032776, 131.7316284
6: -93.8224411, 62.4551201, -93.9196396, 62.5784836, -156.4009247, 156.3747559
7: -65.7589722, 69.1867752, -65.9615021, 69.2981262, -135.0570984, 135.1482849
8: -80.4876862, 83.4292755, -80.6898804, 83.6060333, -164.0937195, 164.1191406
9: -60.2470970, 75.8534851, -60.3677559, 76.1939697, -136.4410706, 136.2212372
10: -87.6998444, 89.3710938, -87.9824524, 89.9872818, -177.6871338, 177.3535461
11: -85.2053757, 57.0813675, -85.5555038, 57.4584694, -142.6638489, 142.6368713
12: -97.0318298, 75.5347061, -97.2012177, 76.0671158, -173.0989380, 172.7359314
13: -84.6063385, 97.7662964, -84.6286926, 98.0516205, -182.6579590, 182.3949890
14: -143.4544678, 80.7228851, -143.6558228, 81.2448730, -224.6993103, 224.3787079
15: -77.7865677, 63.9124794, -78.0716171, 64.0320740, -141.8186340, 141.9841003
16: -90.3308716, 71.3685532, -90.5818787, 71.7464981, -162.0773621, 161.9504242
17: -132.6786346, 70.6407471, -132.9315796, 71.1032562, -203.7818604, 203.5723267
18: -92.7881622, 69.3004379, -93.0050583, 69.3702393, -162.1584015, 162.3054962
19: -67.3196106, 40.0965729, -67.4506836, 40.0763321, -107.3959427, 107.5472565
20: -68.1838531, 52.6357384, -68.3323669, 52.7172775, -120.9011307, 120.9681091
21: -84.6200333, 50.6028023, -84.8133087, 50.6961212, -135.3161621, 135.4161072
22: -85.9075775, 45.9852829, -86.1721344, 46.0902405, -131.9978180, 132.1574097
23: -69.7010040, 53.6800156, -69.8263702, 53.6946449, -123.3956451, 123.5063782
24: -89.6855927, 54.1641846, -89.9436035, 54.1704178, -143.8560028, 144.1077881
25: -75.7961197, 55.0176392, -75.9049072, 55.0781517, -130.8742676, 130.9225464
26: -100.4467392, 81.0987549, -100.5632019, 81.2664795, -181.7132263, 181.6619415
27: -86.9292984, 49.2864456, -87.2586975, 49.3058052, -136.2351074, 136.5451355
28: -68.0799332, 54.1579704, -68.1954575, 54.0959587, -122.1758652, 122.3534241
29: -88.6873550, 41.2476578, -88.8629837, 41.3575439, -130.0448914, 130.1106262
30: -87.9656906, 63.0426865, -88.1121292, 63.2335281, -151.1992188, 151.1548157
31: -90.9879379, 55.6500473, -91.1892090, 55.6406403, -146.6285706, 146.8392639
32: -89.5288239, 56.9162292, -89.6915359, 57.1196823, -146.6484985, 146.6077576
33: -125.7261887, 77.3654327, -126.0876923, 77.5561523, -203.2823334, 203.4531250
34: -105.7433090, 48.2448540, -105.8936920, 48.2863960, -154.0296936, 154.1385498
35: -98.3848114, 58.2679291, -98.6282959, 58.3903275, -156.7751312, 156.8962097
36: -91.9758072, 56.8340683, -92.1287308, 56.9087334, -148.8845367, 148.9627991
37: -144.6468811, 62.2292404, -144.9668121, 62.4293365, -207.0762177, 207.1960449
38: -111.6297607, 70.7133484, -111.8165436, 70.8331375, -182.4628906, 182.5298767
39: -132.4576721, 76.1147308, -132.7369232, 76.3249664, -208.7825928, 208.8516541
40: -110.2388687, 56.5195465, -110.5141068, 56.6362572, -166.8751221, 167.0336609
41: -95.3143387, 65.5765686, -95.4808960, 65.6522217, -160.9665527, 161.0574646
42: -69.9037247, 56.0222969, -70.0331116, 56.2314606, -126.1351852, 126.0553894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=520, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4944718, upper bound: 84.2937406
time: 152.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3489572
time: 124.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -108.1650238, 78.5607224, -108.5175934, 79.0285416, -187.1935730, 187.0783081
1: -56.6849937, 58.8561096, -56.9220314, 59.1507759, -115.8357620, 115.7781372
2: -48.7426262, 60.1997452, -49.0303078, 60.4818115, -109.2244263, 109.2300568
3: -61.3453522, 73.0890732, -61.6370850, 73.4046707, -134.7500305, 134.7261505
4: -63.9236031, 70.3675995, -64.2990112, 70.6883240, -134.6119232, 134.6666107
5: -58.9889717, 72.6793671, -59.3123779, 73.0812225, -132.0701904, 131.9917450
6: -93.8503113, 62.5035515, -94.0454330, 62.6956520, -156.5459595, 156.5489807
7: -65.9198608, 69.2014160, -66.2609634, 69.5272980, -135.4471588, 135.4623718
8: -80.6540070, 83.4469452, -81.0099716, 83.8462677, -164.5002594, 164.4569092
9: -60.3938217, 75.8711700, -60.6577911, 76.4498901, -136.8437195, 136.5289612
10: -87.8364258, 89.4022903, -88.2631531, 90.2276077, -178.0640259, 177.6654358
11: -85.2608948, 57.1401634, -85.7565384, 57.5905380, -142.8514404, 142.8966980
12: -97.0613937, 75.6107712, -97.3330078, 76.2485428, -173.3099365, 172.9437866
13: -84.7177277, 97.8021240, -84.8648911, 98.3325043, -183.0502319, 182.6670227
14: -143.6028290, 80.7475128, -143.9742737, 81.4256744, -225.0285034, 224.7217712
15: -77.8600311, 63.9369202, -78.2430115, 64.1108856, -141.9709167, 142.1799316
16: -90.4992981, 71.3946381, -90.9151077, 72.0069122, -162.5062103, 162.3097534
17: -132.8416748, 70.6705627, -133.2632446, 71.2789459, -204.1206055, 203.9338074
18: -92.8229675, 69.4083328, -93.2359390, 69.5866089, -162.4095764, 162.6442719
19: -67.3511200, 40.1971893, -67.6673126, 40.2617950, -107.6129150, 107.8644943
20: -68.2145233, 52.7323761, -68.5273132, 52.9000854, -121.1146088, 121.2596893
21: -84.6652679, 50.6961136, -85.0427170, 50.8730507, -135.5383148, 135.7388306
22: -85.9465408, 46.0904274, -86.4464645, 46.2892075, -132.2357483, 132.5368958
23: -69.7239380, 53.7868118, -70.0539017, 53.9008560, -123.6247940, 123.8407135
24: -89.7088852, 54.2536812, -90.1822128, 54.3516121, -144.0604706, 144.4358826
25: -75.8224640, 55.1379852, -76.1159973, 55.3079300, -131.1303711, 131.2539825
26: -100.4977112, 81.2499008, -100.8869476, 81.5462875, -182.0440063, 182.1368408
27: -86.9597702, 49.3702621, -87.5481339, 49.4729156, -136.4326782, 136.9183960
28: -68.1077728, 54.2798119, -68.4593048, 54.3296509, -122.4374237, 122.7391129
29: -88.7257919, 41.3229065, -89.1314926, 41.5077972, -130.2335815, 130.4544067
30: -87.9969177, 63.1120911, -88.2830200, 63.3871880, -151.3840942, 151.3951111
31: -91.0207977, 55.7753105, -91.4493103, 55.8805313, -146.9013367, 147.2246246
32: -89.5614166, 56.9578629, -89.8820724, 57.2182312, -146.7796326, 146.8399353
33: -125.7714767, 77.4840851, -126.3415146, 77.8119583, -203.5834198, 203.8255920
34: -105.7705154, 48.3567200, -106.0740204, 48.4969444, -154.2674561, 154.4307404
35: -98.4161682, 58.3825836, -98.8422012, 58.6201172, -157.0362854, 157.2247772
36: -92.0060730, 56.9560051, -92.3672104, 57.1463013, -149.1523743, 149.3232117
37: -144.6968384, 62.3288383, -145.2355957, 62.6317635, -207.3285980, 207.5644379
38: -111.6706314, 70.8461304, -112.1087189, 71.0951843, -182.7658081, 182.9548492
39: -132.5127258, 76.1715393, -132.9644165, 76.4660797, -208.9788055, 209.1359558
40: -110.2877960, 56.5620804, -110.7081070, 56.7342987, -167.0220947, 167.2701721
41: -95.3497086, 65.6199799, -95.6687469, 65.7503204, -161.1000366, 161.2887115
42: -69.9325943, 56.0592384, -70.1446533, 56.3233986, -126.2559814, 126.2038879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=520, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=639, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3322996
time: 167.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3877152
time: 94.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -108.3981628, 78.7425690, -108.3268433, 78.7540894, -187.1522522, 187.0694122
1: -56.8686867, 58.9883842, -56.8128738, 58.9790382, -115.8477173, 115.8012543
2: -49.1820793, 60.3726044, -49.1202011, 60.3446541, -109.5267334, 109.4928055
3: -61.8196220, 73.3125458, -61.7424164, 73.2864075, -135.1060333, 135.0549622
4: -64.4785538, 70.5552673, -64.4086609, 70.5560532, -135.0346069, 134.9639282
5: -59.4125175, 72.9423599, -59.3737679, 72.9356155, -132.3481293, 132.3161163
6: -94.0714264, 62.7515755, -94.0569916, 62.7338791, -156.8052979, 156.8085632
7: -66.2827606, 69.3822021, -66.2306824, 69.3762817, -135.6590424, 135.6128845
8: -81.1149063, 83.6963348, -81.0182037, 83.6940918, -164.8089905, 164.7145233
9: -60.4840088, 76.4559479, -60.4529419, 76.5123672, -136.9963684, 136.9088898
10: -88.1733704, 90.5984650, -88.1130524, 90.6432800, -178.8166504, 178.7115173
11: -85.5979080, 57.9071312, -85.6709595, 57.9004364, -143.4983521, 143.5780945
12: -97.4641876, 76.7973480, -97.3118591, 76.7383575, -174.2025299, 174.1091919
13: -84.8062668, 98.3183289, -84.7182083, 98.3307190, -183.1369934, 183.0365295
14: -143.9249573, 81.8298492, -143.8129272, 81.8421478, -225.7670898, 225.6427765
15: -78.4855499, 64.1523895, -78.4336472, 64.1477737, -142.6333313, 142.5860291
16: -90.6935730, 72.0658951, -90.7333450, 72.1170349, -162.8106079, 162.7992249
17: -133.0422363, 71.5116119, -133.0363007, 71.5571136, -204.5993347, 204.5478973
18: -93.1580124, 69.6063080, -93.1758041, 69.5317078, -162.6896973, 162.7821045
19: -67.5656128, 40.2344055, -67.5586395, 40.1504478, -107.7160492, 107.7930374
20: -68.4277344, 52.9192734, -68.4337387, 52.8697968, -121.2975311, 121.3530045
21: -84.9171371, 50.9428482, -84.9222717, 50.8776093, -135.7947388, 135.8651123
22: -86.3356476, 46.2069435, -86.3893204, 46.1951447, -132.5307922, 132.5962524
23: -69.8928070, 53.8494797, -69.9129944, 53.7831841, -123.6759949, 123.7624588
24: -90.1663742, 54.2748833, -90.1980972, 54.2192383, -144.3856049, 144.4729767
25: -76.0099792, 55.2587318, -76.0124283, 55.2001457, -131.2101288, 131.2711639
26: -100.7438431, 81.7492981, -100.6881180, 81.6194229, -182.3632660, 182.4374084
27: -87.5978165, 49.4382401, -87.6039658, 49.3542557, -136.9520721, 137.0422058
28: -68.3543549, 54.2755394, -68.3318176, 54.1481476, -122.5024796, 122.6073608
29: -88.9530945, 41.5825195, -88.9881592, 41.5338974, -130.4869843, 130.5706787
30: -88.1774063, 63.4617271, -88.2157898, 63.4598999, -151.6372986, 151.6775208
31: -91.3741531, 55.8423729, -91.3729019, 55.7412720, -147.1154175, 147.2152710
32: -89.8105392, 57.3568153, -89.8127441, 57.3557091, -147.1662445, 147.1695557
33: -126.4069061, 77.6455688, -126.4480057, 77.6458435, -204.0527496, 204.0935669
34: -106.1310349, 48.4445953, -106.0974197, 48.3600807, -154.4911041, 154.5420074
35: -98.8965607, 58.4886894, -98.9020996, 58.4528198, -157.3493805, 157.3907928
36: -92.2811279, 56.9467850, -92.2850723, 56.9547997, -149.2359314, 149.2318573
37: -145.0868530, 62.4386368, -145.1878967, 62.5286789, -207.6155396, 207.6265106
38: -111.9867935, 70.9010162, -112.0039597, 70.9171906, -182.9039917, 182.9049683
39: -132.8663177, 76.2565765, -132.9438477, 76.3887482, -209.2550659, 209.2003937
40: -110.7104111, 56.6363678, -110.7466736, 56.6867790, -167.3971863, 167.3830414
41: -95.6094284, 65.7042999, -95.6263428, 65.7166290, -161.3260498, 161.3306274
42: -70.1339493, 56.4905777, -70.1386719, 56.4809341, -126.6148758, 126.6292496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=520, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4954450, upper bound: 84.3805805
time: 157.06 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4954450, upper bound: 84.3805805
time: 120.15 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -108.6131210, 78.7689819, -108.7533264, 79.1186523, -187.7317657, 187.5223083
1: -57.0178833, 59.0018997, -57.0937805, 59.2068253, -116.2247009, 116.0956726
2: -49.2829819, 60.3872871, -49.3146057, 60.5347557, -109.8177338, 109.7018890
3: -61.9277725, 73.3311844, -61.9535141, 73.4993744, -135.4271545, 135.2846985
4: -64.6194534, 70.5707092, -64.6804810, 70.7662354, -135.3856812, 135.2511902
5: -59.5367355, 72.9677124, -59.6086197, 73.1784058, -132.7151489, 132.5763245
6: -94.0994339, 62.8000298, -94.1829529, 62.8510284, -156.9504700, 156.9829865
7: -66.4435120, 69.3968658, -66.5300369, 69.6054535, -136.0489502, 135.9269104
8: -81.2810745, 83.7140427, -81.3381195, 83.9343109, -165.2153931, 165.0521545
9: -60.6307831, 76.4736633, -60.7429962, 76.7683258, -137.3991089, 137.2166595
10: -88.3099518, 90.6296692, -88.3938904, 90.8836670, -179.1936188, 179.0235596
11: -85.6535721, 57.9659233, -85.8724899, 58.0324860, -143.6860504, 143.8384094
12: -97.4938278, 76.8732605, -97.4436951, 76.9196167, -174.4134521, 174.3169556
13: -84.9176331, 98.3541412, -84.9544449, 98.6116028, -183.5292358, 183.3085632
14: -144.0733948, 81.8543701, -144.1315308, 82.0229034, -226.0962524, 225.9859009
15: -78.5589905, 64.1769333, -78.6049805, 64.2266159, -142.7856140, 142.7819061
16: -90.8620224, 72.0920258, -91.0665359, 72.3774796, -163.2395020, 163.1585541
17: -133.2052612, 71.5413208, -133.3680878, 71.7329102, -204.9381714, 204.9094086
18: -93.1928711, 69.7142334, -93.4066696, 69.7481537, -162.9410095, 163.1209106
19: -67.5972214, 40.3350410, -67.7754211, 40.3359108, -107.9331360, 108.1104584
20: -68.4585571, 53.0159492, -68.6287994, 53.0526276, -121.5111847, 121.6447372
21: -84.9625015, 51.0361862, -85.1518402, 51.0545540, -136.0170593, 136.1880188
22: -86.3745956, 46.3120575, -86.6636429, 46.3940773, -132.7686615, 132.9757080
23: -69.9157562, 53.9562683, -70.1405716, 53.9894333, -123.9051895, 124.0968399
24: -90.1897354, 54.3643684, -90.4367752, 54.4004898, -144.5902100, 144.8011322
25: -76.0363312, 55.3789978, -76.2235489, 55.4299164, -131.4662476, 131.6025391
26: -100.7949829, 81.9004288, -101.0118637, 81.8992081, -182.6941833, 182.9122925
27: -87.6281662, 49.5220795, -87.8934402, 49.5214119, -137.1495819, 137.4155273
28: -68.3821411, 54.3973694, -68.5956573, 54.3818817, -122.7640152, 122.9930191
29: -88.9914856, 41.6577225, -89.2566986, 41.6840820, -130.6755676, 130.9144287
30: -88.2087326, 63.5310745, -88.3867874, 63.6135788, -151.8222961, 151.9178619
31: -91.4070587, 55.9676437, -91.6330566, 55.9812164, -147.3882599, 147.6007080
32: -89.8431854, 57.3983955, -90.0034256, 57.4542046, -147.2973785, 147.4018250
33: -126.4522781, 77.7642899, -126.7019348, 77.9016800, -204.3539429, 204.4662170
34: -106.1583099, 48.5563698, -106.2777939, 48.5706253, -154.7289276, 154.8341675
35: -98.9278870, 58.6033936, -99.1161652, 58.6826324, -157.6105042, 157.7195587
36: -92.3114014, 57.0686874, -92.5235672, 57.1922836, -149.5036926, 149.5922546
37: -145.1368561, 62.5382385, -145.4566956, 62.7311401, -207.8679962, 207.9949341
38: -112.0276947, 71.0338440, -112.2962112, 71.1792068, -183.2069092, 183.3300476
39: -132.9213562, 76.3134003, -133.1712646, 76.5299149, -209.4512482, 209.4846497
40: -110.7593155, 56.6788864, -110.9408188, 56.7848969, -167.5442200, 167.6197052
41: -95.6448593, 65.7476959, -95.8143387, 65.8147812, -161.4596405, 161.5620422
42: -70.1629486, 56.5275078, -70.2503662, 56.5728531, -126.7358017, 126.7778778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=520, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=639, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4954450, upper bound: 84.4191954
time: 96.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4954450, upper bound: 84.4191954
time: 147.27 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -107.9797897, 78.5535278, -108.2095337, 78.7378006, -186.7175751, 186.7630615
1: -56.5510216, 58.8606873, -56.7082481, 59.0036316, -115.5546494, 115.5689240
2: -48.6873283, 60.2049980, -48.9461899, 60.3984947, -109.0858231, 109.1511841
3: -61.3018913, 73.0937653, -61.5744667, 73.3464966, -134.6483765, 134.6682281
4: -63.8134880, 70.3749542, -64.1247711, 70.6033936, -134.4168701, 134.4997253
5: -58.9024048, 72.6770935, -59.1796722, 72.9656372, -131.8680420, 131.8567657
6: -93.9179459, 62.4699326, -94.1430435, 62.6871490, -156.6050873, 156.6129761
7: -65.7991943, 69.2068787, -66.0845490, 69.4042053, -135.2033997, 135.2914276
8: -80.5155792, 83.4528046, -80.7839813, 83.7216339, -164.2371979, 164.2367859
9: -60.2712784, 75.9672394, -60.6113815, 76.4116058, -136.6828918, 136.5786133
10: -87.7318115, 89.6060028, -88.4755707, 90.4265137, -178.1583252, 178.0815735
11: -85.2312164, 57.1446457, -85.7967606, 57.5907555, -142.8219604, 142.9414062
12: -97.0639191, 75.6274414, -97.5038147, 76.2599106, -173.3238068, 173.1312561
13: -84.6429596, 97.8008423, -84.7841339, 98.1755295, -182.8184814, 182.5849762
14: -143.4970245, 80.8890381, -144.0507812, 81.5624390, -225.0594635, 224.9398193
15: -77.8176270, 63.9562378, -78.1933365, 64.1665039, -141.9841309, 142.1495667
16: -90.3691177, 71.4751129, -90.8521194, 71.9651947, -162.3343201, 162.3272400
17: -132.7073364, 70.6871338, -133.1786346, 71.2333527, -203.9406738, 203.8657684
18: -92.8194733, 69.3388214, -93.1511307, 69.4943542, -162.3138275, 162.4899445
19: -67.3512268, 40.1229477, -67.5963440, 40.1517639, -107.5029831, 107.7192917
20: -68.2103119, 52.6483536, -68.4434509, 52.7690430, -120.9793396, 121.0917969
21: -84.6547165, 50.6498299, -85.0160904, 50.8078766, -135.4625854, 135.6659241
22: -85.9536743, 46.0118561, -86.3223343, 46.1779022, -132.1315613, 132.3341980
23: -69.7256317, 53.7027664, -69.9444122, 53.7649155, -123.4905472, 123.6471710
24: -89.7268219, 54.1789627, -90.0522766, 54.2685661, -143.9953766, 144.2312317
25: -75.8214111, 55.0416946, -76.0077972, 55.1733170, -130.9947205, 131.0494690
26: -100.4793549, 81.1757278, -100.7667770, 81.4470444, -181.9263916, 181.9425049
27: -87.0149918, 49.3003616, -87.4588394, 49.4249802, -136.4399719, 136.7592010
28: -68.1325684, 54.1744194, -68.3262329, 54.2100220, -122.3425903, 122.5006409
29: -88.7292099, 41.2873497, -89.0064316, 41.4472351, -130.1764221, 130.2937775
30: -87.9944229, 63.0714417, -88.2195282, 63.3225174, -151.3169250, 151.2909698
31: -91.0252075, 55.6737213, -91.3401108, 55.7122116, -146.7374268, 147.0138245
32: -89.5843964, 56.9269676, -89.8417358, 57.1810951, -146.7654724, 146.7687073
33: -125.8471832, 77.3871384, -126.3234634, 77.8476562, -203.6948395, 203.7106018
34: -105.8626099, 48.2651482, -106.1318283, 48.5209274, -154.3835449, 154.3969727
35: -98.4925079, 58.2840576, -98.8404999, 58.6692467, -157.1617584, 157.1245575
36: -92.0852356, 56.8446808, -92.3590851, 57.0914536, -149.1766968, 149.2037659
37: -144.7253113, 62.2514343, -145.1690063, 62.6015701, -207.3268738, 207.4204407
38: -111.7641602, 70.7362823, -112.0962677, 71.0954285, -182.8595734, 182.8325500
39: -132.5344238, 76.1318130, -132.9363556, 76.5143585, -209.0487823, 209.0681763
40: -110.3272858, 56.5301208, -110.7194672, 56.7941666, -167.1214447, 167.2495880
41: -95.3941345, 65.5915833, -95.6689758, 65.7704391, -161.1645813, 161.2605591
42: -69.9480133, 56.0447617, -70.1764450, 56.3250542, -126.2730560, 126.2212067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=520, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4944718, upper bound: 84.2937406
time: 105.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4944718, upper bound: 84.2937406
time: 90.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -108.1947479, 78.5798492, -108.6360779, 79.1022568, -187.2969971, 187.2159271
1: -56.7002754, 58.8742294, -56.9892426, 59.2314491, -115.9317245, 115.8634720
2: -48.7882423, 60.2196808, -49.1405792, 60.5886040, -109.3768311, 109.3602600
3: -61.4100113, 73.1123810, -61.7855835, 73.5594559, -134.9694519, 134.8979492
4: -63.9546013, 70.3904114, -64.3967590, 70.8135529, -134.7681580, 134.7871704
5: -59.0265846, 72.7022476, -59.4146233, 73.2083511, -132.2349243, 132.1168671
6: -93.9458389, 62.5184364, -94.2687988, 62.8043213, -156.7501526, 156.7872314
7: -65.9600067, 69.2215424, -66.3839722, 69.6334534, -135.5934601, 135.6055145
8: -80.6818619, 83.4705048, -81.1040344, 83.9618530, -164.6437073, 164.5745392
9: -60.4180527, 75.9849396, -60.9014320, 76.6674957, -137.0855408, 136.8863678
10: -87.8683624, 89.6372070, -88.7563629, 90.6668396, -178.5351868, 178.3935547
11: -85.2867050, 57.2034302, -85.9977417, 57.7228127, -143.0095215, 143.2011566
12: -97.0935822, 75.7035370, -97.6357422, 76.4413223, -173.5348816, 173.3392792
13: -84.7543030, 97.8367157, -85.0203781, 98.4563370, -183.2106323, 182.8570862
14: -143.6454010, 80.9136581, -144.3692017, 81.7431717, -225.3885803, 225.2828674
15: -77.8911133, 63.9807167, -78.3647842, 64.2453003, -142.1364136, 142.3454895
16: -90.5375290, 71.5012054, -91.1853714, 72.2256088, -162.7631378, 162.6865692
17: -132.8703156, 70.7169800, -133.5104370, 71.4089966, -204.2793121, 204.2274170
18: -92.8543167, 69.4467087, -93.3819656, 69.7107086, -162.5650330, 162.8286743
19: -67.3827591, 40.2235718, -67.8129730, 40.3372154, -107.7199554, 108.0365372
20: -68.2410278, 52.7450142, -68.6383514, 52.9518547, -121.1928864, 121.3833618
21: -84.6999817, 50.7431564, -85.2454605, 50.9848633, -135.6848297, 135.9886169
22: -85.9926224, 46.1170082, -86.5966644, 46.3768883, -132.3695068, 132.7136688
23: -69.7485428, 53.8095703, -70.1719513, 53.9711189, -123.7196655, 123.9815216
24: -89.7501831, 54.2684555, -90.2908554, 54.4497757, -144.1999512, 144.5593109
25: -75.8477707, 55.1620026, -76.2189026, 55.4031105, -131.2508850, 131.3809052
26: -100.5302887, 81.3268814, -101.0905304, 81.7268677, -182.2571411, 182.4173889
27: -87.0454102, 49.3842201, -87.7481918, 49.5920868, -136.6374969, 137.1324158
28: -68.1603546, 54.2962570, -68.5899963, 54.4437370, -122.6040955, 122.8862534
29: -88.7676163, 41.3626099, -89.2749405, 41.5974808, -130.3650970, 130.6375427
30: -88.0256729, 63.1408005, -88.3904190, 63.4761925, -151.5018616, 151.5312195
31: -91.0580750, 55.7989883, -91.6002655, 55.9520721, -147.0101471, 147.3992615
32: -89.6169891, 56.9686165, -90.0322952, 57.2796249, -146.8966064, 147.0009155
33: -125.8925247, 77.5058517, -126.5772705, 78.1035004, -203.9960327, 204.0831299
34: -105.8898087, 48.3769264, -106.3121567, 48.7314301, -154.6212463, 154.6890869
35: -98.5238495, 58.3987198, -99.0543747, 58.8990707, -157.4228973, 157.4530945
36: -92.1155243, 56.9666061, -92.5975037, 57.3289337, -149.4444580, 149.5641174
37: -144.7753143, 62.3509979, -145.4377136, 62.8039932, -207.5793152, 207.7886963
38: -111.8050842, 70.8690948, -112.3883896, 71.3575134, -183.1625977, 183.2574768
39: -132.5894928, 76.1885986, -133.1637573, 76.6554489, -209.2449341, 209.3523560
40: -110.3762207, 56.5726433, -110.9134140, 56.8922501, -167.2684631, 167.4860535
41: -95.4294891, 65.6350098, -95.8568039, 65.8685608, -161.2980499, 161.4918213
42: -69.9768906, 56.0817108, -70.2879639, 56.4170113, -126.3939056, 126.3696594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=520, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=639, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3322996
time: 158.16 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3877152
time: 114.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -108.4277802, 78.7614746, -108.4449310, 78.8279266, -187.2556915, 187.2063904
1: -56.8839417, 59.0061188, -56.8798752, 59.0597458, -115.9436874, 115.8859940
2: -49.2276154, 60.3924217, -49.2304611, 60.4512520, -109.6788635, 109.6228790
3: -61.8834190, 73.3355637, -61.8903885, 73.4408417, -135.3242493, 135.2259521
4: -64.5097351, 70.5776825, -64.5061798, 70.6809769, -135.1907043, 135.0838623
5: -59.4489899, 72.9647675, -59.4750137, 73.0623322, -132.5113220, 132.4397736
6: -94.1667328, 62.7665787, -94.2807770, 62.8425789, -157.0092926, 157.0473633
7: -66.3231659, 69.4018097, -66.3539047, 69.4819107, -135.8050690, 135.7557068
8: -81.1428070, 83.7196350, -81.1123810, 83.8094177, -164.9522095, 164.8320160
9: -60.5078888, 76.5693359, -60.6961823, 76.7297821, -137.2376709, 137.2655182
10: -88.2047348, 90.8330231, -88.6055603, 91.0820541, -179.2867889, 179.4385834
11: -85.6235504, 57.9702950, -85.9119415, 58.0326080, -143.6561584, 143.8822327
12: -97.4959412, 76.8898087, -97.6137848, 76.9312592, -174.4272003, 174.5035706
13: -84.8429947, 98.3529968, -84.8735733, 98.4544525, -183.2974548, 183.2265625
14: -143.9671936, 81.9959564, -144.2073212, 82.1596985, -226.1268921, 226.2032776
15: -78.5165558, 64.1961212, -78.5549545, 64.2827301, -142.7992859, 142.7510681
16: -90.7312546, 72.1722870, -91.0029449, 72.3354950, -163.0667419, 163.1752319
17: -133.0707703, 71.5582581, -133.2831879, 71.6871185, -204.7578583, 204.8414459
18: -93.1893539, 69.6445999, -93.3219070, 69.6557312, -162.8450775, 162.9665070
19: -67.5969391, 40.2602654, -67.7041626, 40.2252655, -107.8221893, 107.9644318
20: -68.4539413, 52.9319305, -68.5449524, 52.9214325, -121.3753510, 121.4768829
21: -84.9514999, 50.9894218, -85.1248398, 50.9891663, -135.9406738, 136.1142578
22: -86.3821716, 46.2335663, -86.5395889, 46.2827377, -132.6649017, 132.7731628
23: -69.9171906, 53.8723755, -70.0308151, 53.8534775, -123.7706451, 123.9031830
24: -90.2077255, 54.2897873, -90.3069153, 54.3173370, -144.5250549, 144.5967102
25: -76.0354462, 55.2826843, -76.1152191, 55.2949791, -131.3304291, 131.3979034
26: -100.7762375, 81.8258057, -100.8910141, 81.7996902, -182.5759125, 182.7168274
27: -87.6838074, 49.4520950, -87.8043747, 49.4733772, -137.1571808, 137.2564697
28: -68.4071884, 54.2923927, -68.4627609, 54.2623901, -122.6695709, 122.7551498
29: -88.9948425, 41.6223221, -89.1314926, 41.6235123, -130.6183319, 130.7538147
30: -88.2056503, 63.4915085, -88.3231735, 63.5487404, -151.7543793, 151.8146820
31: -91.4114151, 55.8656120, -91.5234604, 55.8126640, -147.2240753, 147.3890686
32: -89.8658752, 57.3676491, -89.9632950, 57.4170647, -147.2829285, 147.3309479
33: -126.5280380, 77.6670685, -126.6838303, 77.9371109, -204.4651489, 204.3508911
34: -106.2504120, 48.4644775, -106.3356857, 48.5942764, -154.8446960, 154.8001709
35: -99.0042343, 58.5045471, -99.1142807, 58.7314224, -157.7356415, 157.6188202
36: -92.3905640, 56.9572372, -92.5154419, 57.1372604, -149.5278168, 149.4726868
37: -145.1655273, 62.4607620, -145.3902893, 62.7006798, -207.8662109, 207.8510437
38: -112.1206589, 70.9236755, -112.2832565, 71.1790466, -183.2997131, 183.2069397
39: -132.9432983, 76.2735291, -133.1435699, 76.5780640, -209.5213623, 209.4170990
40: -110.7991486, 56.6469421, -110.9520645, 56.8447342, -167.6438904, 167.5989990
41: -95.6892014, 65.7192383, -95.8146744, 65.8348846, -161.5240784, 161.5339050
42: -70.1779556, 56.5133438, -70.2821732, 56.5741730, -126.7521286, 126.7955170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=520, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5578587, upper bound: 84.3805805
time: 108.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4944718, upper bound: 84.4368157
time: 93.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -108.6427460, 78.7878952, -108.8714142, 79.1924362, -187.8351746, 187.6593018
1: -57.0331039, 59.0196190, -57.1607933, 59.2875519, -116.3206558, 116.1804123
2: -49.3285103, 60.4070969, -49.4248314, 60.6413193, -109.9698257, 109.8319244
3: -61.9915237, 73.3542480, -62.1014900, 73.6538239, -135.6453400, 135.4557343
4: -64.6506500, 70.5931473, -64.7780304, 70.8911743, -135.5418091, 135.3711700
5: -59.5731926, 72.9901276, -59.7099113, 73.3050919, -132.8782806, 132.7000427
6: -94.1947250, 62.8150215, -94.4066696, 62.9597549, -157.1544647, 157.2216949
7: -66.4838715, 69.4165039, -66.6532135, 69.7111130, -136.1949768, 136.0697174
8: -81.3089447, 83.7373047, -81.4323120, 84.0496216, -165.3585663, 165.1695862
9: -60.6546898, 76.5870361, -60.9862671, 76.9857025, -137.6403809, 137.5733032
10: -88.3413849, 90.8641891, -88.8864441, 91.3224487, -179.6637878, 179.7506104
11: -85.6792068, 58.0290527, -86.1133728, 58.1646881, -143.8439026, 144.1424255
12: -97.5255585, 76.9656982, -97.7456512, 77.1124954, -174.6380615, 174.7113342
13: -84.9543381, 98.3887787, -85.1098480, 98.7353134, -183.6896515, 183.4986267
14: -144.1156464, 82.0205002, -144.5259705, 82.3404465, -226.4560852, 226.5464783
15: -78.5899658, 64.2206879, -78.7263184, 64.3616028, -142.9515533, 142.9470062
16: -90.8997421, 72.1984253, -91.3361740, 72.5958786, -163.4956207, 163.5345764
17: -133.2338257, 71.5879898, -133.6150818, 71.8628616, -205.0966797, 205.2030487
18: -93.2242737, 69.7524948, -93.5527267, 69.8721390, -163.0964050, 163.3052216
19: -67.6285858, 40.3608780, -67.9209290, 40.4107323, -108.0393219, 108.2818069
20: -68.4847641, 53.0285873, -68.7399750, 53.1042442, -121.5889816, 121.7685623
21: -84.9969177, 51.0827522, -85.3543625, 51.1661682, -136.1630859, 136.4371033
22: -86.4210587, 46.3386841, -86.8139191, 46.4816933, -132.9027557, 133.1526031
23: -69.9401627, 53.9792023, -70.2583923, 54.0597115, -123.9998627, 124.2375793
24: -90.2310791, 54.3792915, -90.5455475, 54.4985847, -144.7296600, 144.9248352
25: -76.0617981, 55.4029770, -76.3263550, 55.5247536, -131.5865479, 131.7293091
26: -100.8273621, 81.9769440, -101.2147827, 82.0795593, -182.9068909, 183.1917267
27: -87.7141113, 49.5359344, -88.0937881, 49.6405334, -137.3546448, 137.6297302
28: -68.4349365, 54.4142113, -68.7265244, 54.4961166, -122.9310455, 123.1407318
29: -89.0332489, 41.6975136, -89.4000015, 41.7737656, -130.8070068, 131.0975037
30: -88.2370148, 63.5608940, -88.4941788, 63.7024918, -151.9395142, 152.0550537
31: -91.4443512, 55.9908714, -91.7836075, 56.0525970, -147.4969482, 147.7744751
32: -89.8985672, 57.4092369, -90.1539230, 57.5155945, -147.4141541, 147.5631561
33: -126.5733185, 77.7858353, -126.9377213, 78.1929932, -204.7663116, 204.7235413
34: -106.2776947, 48.5763474, -106.5160446, 48.8048363, -155.0825043, 155.0923767
35: -99.0355682, 58.6192436, -99.3282623, 58.9612617, -157.9968262, 157.9475098
36: -92.4208298, 57.0791473, -92.7539062, 57.3747940, -149.7956238, 149.8330536
37: -145.2155457, 62.5603256, -145.6591034, 62.9031601, -208.1187134, 208.2194214
38: -112.1615906, 71.0565186, -112.5754547, 71.4411774, -183.6027679, 183.6319733
39: -132.9983521, 76.3303909, -133.3709717, 76.7191772, -209.7175293, 209.7013550
40: -110.8480377, 56.6894569, -111.1461411, 56.9428482, -167.7908630, 167.8356018
41: -95.7246246, 65.7626343, -96.0026627, 65.9329987, -161.6576233, 161.7652893
42: -70.2068939, 56.5502701, -70.3938293, 56.6661110, -126.8729858, 126.9440918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=520, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=639, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5578587, upper bound: 84.4191954
time: 109.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4944718, upper bound: 84.4756721
time: 127.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -108.1709518, 78.6699982, -108.1453018, 78.7211838, -186.8921356, 186.8152924
1: -56.6635971, 58.9387932, -56.6633377, 58.9567375, -115.6203308, 115.6021271
2: -48.7408295, 60.2814140, -48.8664207, 60.3205795, -109.0614090, 109.1478348
3: -61.3602180, 73.2452087, -61.4613304, 73.2629929, -134.6232147, 134.7065430
4: -63.9145775, 70.5168457, -64.0744019, 70.5299988, -134.4445801, 134.5912476
5: -59.0172272, 72.8382797, -59.1088905, 72.9237366, -131.9409637, 131.9471741
6: -94.0449295, 62.5703850, -94.0120544, 62.5992393, -156.6441650, 156.5824432
7: -65.9412231, 69.3419342, -65.9910583, 69.3755341, -135.3167572, 135.3329773
8: -80.5996094, 83.5734024, -80.7180328, 83.6498184, -164.2494202, 164.2914429
9: -60.5687561, 76.1993103, -60.3991165, 76.3760376, -136.9447937, 136.5984192
10: -88.2356949, 89.9158478, -88.0227890, 90.2788544, -178.5145569, 177.9386292
11: -85.5768509, 57.2694855, -85.6000900, 57.5514183, -143.1282654, 142.8695679
12: -97.2314758, 75.6672440, -97.2676773, 76.1016312, -173.3330994, 172.9349060
13: -84.7566071, 97.9329376, -84.6716537, 98.0962830, -182.8528900, 182.6045837
14: -143.8456726, 81.0913696, -143.7070007, 81.4439087, -225.2895813, 224.7983704
15: -77.9011765, 64.0762482, -78.1018829, 64.0881348, -141.9893036, 142.1781311
16: -90.7698822, 71.7287292, -90.6361923, 71.9387054, -162.7085876, 162.3649292
17: -133.0558472, 70.8526230, -132.9850159, 71.1664200, -204.2222595, 203.8376465
18: -92.9703293, 69.4255371, -93.0523911, 69.4165955, -162.3869324, 162.4779205
19: -67.4924927, 40.1543007, -67.5000458, 40.0966492, -107.5891418, 107.6543427
20: -68.3330994, 52.7238693, -68.3697968, 52.7509918, -121.0840912, 121.0936661
21: -84.8806458, 50.7110901, -84.8702621, 50.7392273, -135.6198730, 135.5813599
22: -86.1959381, 46.1447182, -86.2931061, 46.1223984, -132.3183289, 132.4378204
23: -69.8728485, 53.7611046, -69.8692169, 53.7265320, -123.5993805, 123.6303101
24: -89.8807602, 54.2755127, -90.0332565, 54.1927948, -144.0735474, 144.3087769
25: -75.9356842, 55.1446457, -75.9503250, 55.1098289, -131.0455017, 131.0949707
26: -100.6395950, 81.2203064, -100.6221008, 81.3078918, -181.9474792, 181.8424072
27: -87.1473846, 49.3865051, -87.3497086, 49.3282166, -136.4756012, 136.7362061
28: -68.1924591, 54.2375908, -68.2333527, 54.1193619, -122.3118134, 122.4709473
29: -88.9198761, 41.3293953, -88.9389801, 41.3822403, -130.3021240, 130.2683716
30: -88.1486893, 63.2134399, -88.1546783, 63.3078499, -151.4565277, 151.3681030
31: -91.1669388, 55.7063408, -91.2443542, 55.6622734, -146.8291931, 146.9506989
32: -89.7311630, 57.0079193, -89.7785187, 57.1366730, -146.8678284, 146.7864380
33: -126.1079254, 77.7390518, -126.2947617, 77.5777130, -203.6856232, 204.0338135
34: -105.9427185, 48.4760742, -105.9946976, 48.3121643, -154.2548828, 154.4707642
35: -98.6387939, 58.5930290, -98.7599335, 58.4089470, -157.0477142, 157.3529663
36: -92.2596664, 57.1140213, -92.2747650, 56.9245148, -149.1841736, 149.3887939
37: -145.1451111, 62.5577507, -145.2195129, 62.4515152, -207.5966187, 207.7772675
38: -111.9918594, 71.0916672, -111.9954453, 70.8634644, -182.8553162, 183.0871124
39: -132.9029846, 76.4777069, -132.9595490, 76.3434601, -209.2464447, 209.4372559
40: -110.5827332, 56.7653999, -110.6782150, 56.6521759, -167.2349091, 167.4436188
41: -95.5950394, 65.7395325, -95.6181717, 65.6671982, -161.2622375, 161.3576965
42: -70.1031647, 56.1040344, -70.1104736, 56.2480202, -126.3511658, 126.2145081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=520, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 919
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 920
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1414
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 746
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 934
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1684
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1470
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 922
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1029
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1028
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 921
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1626
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1034
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1528
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1031
type: A, layer: 1, pos: 1038
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1030
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1033
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1027
type: A, layer: 1, pos: 1249
type: A, layer: 1, pos: 1037
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1035
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1032
type: A, layer: 1, pos: 1039
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1232
type: A, layer: 1, pos: 1535
type: A, layer: 1, pos: 791
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1248
type: A, layer: 1, pos: 1264
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1688

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3777200
time: 1029.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3777200
time: 113.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 1145.92 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4944718, upper bound: 84.2937406
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3489572
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3322996
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3877152
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4954450, upper bound: 84.3805805
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4954450, upper bound: 84.3805805
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4954450, upper bound: 84.4191954
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4954450, upper bound: 84.4191954
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4944718, upper bound: 84.2937406
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4944718, upper bound: 84.2937406
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3322996
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3877152
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.5578587, upper bound: 84.3805805
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4944718, upper bound: 84.4368157
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.5578587, upper bound: 84.4191954
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4944718, upper bound: 84.4756721
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3777200
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1145.92
Output dim: 9, lower bound: -84.4944718, upper bound: 84.3777200
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1145.92
Output dim: 9, lower bound: -84.4989034, upper bound: 84.4747943
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1145.92
Output dim: 9, lower bound: -84.4993118, upper bound: 84.5230785
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1145.92
Output dim: 9, lower bound: -84.4993118, upper bound: 84.5618419
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1145.92
Output dim: 9, lower bound: -84.4989034, upper bound: 84.4360997
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1145.92
Output dim: 9, lower bound: -84.4989034, upper bound: 84.4747943
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1145.92
Output dim: 9, lower bound: -84.4993118, upper bound: 84.5230785
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1145.92
Output dim: 9, lower bound: -84.4993118, upper bound: 84.5618419

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 129.90 + 7972.62 = 8102.52 seconds
