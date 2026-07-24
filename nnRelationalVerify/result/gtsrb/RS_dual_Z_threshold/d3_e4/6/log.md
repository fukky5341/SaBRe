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
execution time: IAR + RelationalAnalysis = 2.90 + 127.44 = 130.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -84.5784015, upper bound: 84.5784015

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5341717, upper bound: 84.5728647
time: 101.83 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5728647, upper bound: 84.5341717
time: 110.09 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 212.06 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 212.06
Output dim: 9, lower bound: -84.5341717, upper bound: 84.5728647
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 212.06
Output dim: 9, lower bound: -84.5728647, upper bound: 84.5341717

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4913851, upper bound: 84.5703286
time: 115.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5317018, upper bound: 84.5313914
time: 93.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5313914, upper bound: 84.5317018
time: 115.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5703286, upper bound: 84.4913851
time: 175.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 293.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 293.27
Output dim: 9, lower bound: -84.4913851, upper bound: 84.5703286
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 293.27
Output dim: 9, lower bound: -84.5317018, upper bound: 84.5313914
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 293.27
Output dim: 9, lower bound: -84.5313914, upper bound: 84.5317018
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 293.27
Output dim: 9, lower bound: -84.5703286, upper bound: 84.4913851

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1685

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -84.4856147, upper bound: 84.4841346
time: 217.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4051014, upper bound: 84.5646268
time: 98.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1685

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5259679, upper bound: 84.4452028
time: 109.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4454568, upper bound: 84.5256973
time: 111.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1685

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5256973, upper bound: 84.4454568
time: 130.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4452027, upper bound: 84.5259679
time: 143.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1685

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5646268, upper bound: 84.4051014
time: 342.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -84.4841346, upper bound: 84.4856147
time: 121.26 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 470.45 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 470.45
Output dim: 9, lower bound: -84.4856147, upper bound: 84.4841346
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 470.45
Output dim: 9, lower bound: -84.4051014, upper bound: 84.5646268
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 470.45
Output dim: 9, lower bound: -84.5259679, upper bound: 84.4452028
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 470.45
Output dim: 9, lower bound: -84.4454568, upper bound: 84.5256973
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 470.45
Output dim: 9, lower bound: -84.5256973, upper bound: 84.4454568
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 470.45
Output dim: 9, lower bound: -84.4452027, upper bound: 84.5259679
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 470.45
Output dim: 9, lower bound: -84.5646268, upper bound: 84.4051014
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 470.45
Output dim: 9, lower bound: -84.4841346, upper bound: 84.4856147

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1683

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4002091, upper bound: 84.5069988
time: 107.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3470286, upper bound: 84.5597722
time: 124.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1683

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5211389, upper bound: 84.3871770
time: 112.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -84.4682839, upper bound: 84.4403044
time: 104.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1683

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -84.4405666, upper bound: 84.4680322
time: 116.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3874117, upper bound: 84.5208511
time: 98.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1683

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5208511, upper bound: 84.3874117
time: 99.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -84.4680322, upper bound: 84.4405666
time: 103.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1683

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -84.4403044, upper bound: 84.4682839
time: 95.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3871770, upper bound: 84.5211389
time: 184.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1683

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5597722, upper bound: 84.3470286
time: 114.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5069988, upper bound: 84.4002091
time: 122.10 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 244.02 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 244.02
Output dim: 9, lower bound: -84.4002091, upper bound: 84.5069988
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 244.02
Output dim: 9, lower bound: -84.3470286, upper bound: 84.5597722
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 244.02
Output dim: 9, lower bound: -84.5211389, upper bound: 84.3871770
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 244.02
Output dim: 9, lower bound: -84.4682839, upper bound: 84.4403044
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 244.02
Output dim: 9, lower bound: -84.4405666, upper bound: 84.4680322
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 244.02
Output dim: 9, lower bound: -84.3874117, upper bound: 84.5208511
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 244.02
Output dim: 9, lower bound: -84.5208511, upper bound: 84.3874117
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 244.02
Output dim: 9, lower bound: -84.4680322, upper bound: 84.4405666
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 244.02
Output dim: 9, lower bound: -84.4403044, upper bound: 84.4682839
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 244.02
Output dim: 9, lower bound: -84.3871770, upper bound: 84.5211389
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 244.02
Output dim: 9, lower bound: -84.5597722, upper bound: 84.3470286
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 244.02
Output dim: 9, lower bound: -84.5069988, upper bound: 84.4002091

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3781033, upper bound: 84.5038856
time: 161.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -84.3971137, upper bound: 84.4849433
time: 114.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3248555, upper bound: 84.5566676
time: 100.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3439274, upper bound: 84.5377965
time: 122.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4991096, upper bound: 84.3840700
time: 184.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5180330, upper bound: 84.3649605
time: 105.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3652172, upper bound: 84.5177432
time: 97.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3843034, upper bound: 84.4988494
time: 144.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4988494, upper bound: 84.3843034
time: 139.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5177432, upper bound: 84.3652172
time: 109.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3649604, upper bound: 84.5180330
time: 143.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3840700, upper bound: 84.4991097
time: 106.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5377965, upper bound: 84.3439274
time: 108.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5566676, upper bound: 84.3248555
time: 121.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -84.4849433, upper bound: 84.3971137
time: 135.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5038856, upper bound: 84.3781033
time: 100.05 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 242.82 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.3781033, upper bound: 84.5038856
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.3971137, upper bound: 84.4849433
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.3248555, upper bound: 84.5566676
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.3439274, upper bound: 84.5377965
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.4991096, upper bound: 84.3840700
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.5180330, upper bound: 84.3649605
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.3652172, upper bound: 84.5177432
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.3843034, upper bound: 84.4988494
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.4988494, upper bound: 84.3843034
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.5177432, upper bound: 84.3652172
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.3649604, upper bound: 84.5180330
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.3840700, upper bound: 84.4991097
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.5377965, upper bound: 84.3439274
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.5566676, upper bound: 84.3248555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.4849433, upper bound: 84.3971137
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 242.82
Output dim: 9, lower bound: -84.5038856, upper bound: 84.3781033

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3382305, upper bound: 84.5013587
time: 115.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -84.3755522, upper bound: 84.4639984
time: 94.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.2849676, upper bound: 84.5541317
time: 120.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3222989, upper bound: 84.5167747
time: 104.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3041106, upper bound: 84.5352674
time: 132.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.3413951, upper bound: 84.4978500
time: 102.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -84.4592338, upper bound: 84.3815382
time: 95.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.4965516, upper bound: 84.3441573
time: 111.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -108.9149094, 78.9572983, -108.9149094, 78.9572983, -187.8722076, 187.8722076
1: -57.1994133, 59.1304626, -57.1994133, 59.1304626, -116.3298798, 116.3298798
2: -49.4722519, 60.5177383, -49.4722519, 60.5177383, -109.9899902, 109.9899902
3: -62.1584549, 73.5739899, -62.1584549, 73.5739899, -135.7324524, 135.7324524
4: -64.8608780, 70.7777328, -64.8608780, 70.7777328, -135.6386108, 135.6386108
5: -59.7658234, 73.2126312, -59.7658234, 73.2126312, -132.9784393, 132.9784546
6: -94.4622650, 62.9764252, -94.4622650, 62.9764252, -157.4386902, 157.4386902
7: -66.7086945, 69.5976562, -66.7086945, 69.5976562, -136.3063354, 136.3063507
8: -81.4671783, 83.9015045, -81.4671783, 83.9015045, -165.3686829, 165.3686829
9: -61.0049210, 76.9824524, -61.0049210, 76.9824524, -137.9873657, 137.9873657
10: -88.9067383, 91.4983521, -88.9067383, 91.4983521, -180.4050903, 180.4050903
11: -86.0778122, 58.2743568, -86.0778122, 58.2743568, -144.3521729, 144.3521729
12: -97.7510147, 77.1579056, -97.7510147, 77.1579056, -174.9089203, 174.9089203
13: -85.1460876, 98.5895386, -85.1460876, 98.5895386, -183.7356262, 183.7356262
14: -144.5496368, 82.4510651, -144.5496368, 82.4510651, -227.0007019, 227.0006866
15: -78.7490540, 64.4168396, -78.7490540, 64.4168396, -143.1658936, 143.1658936
16: -91.3781662, 72.6167374, -91.3781662, 72.6167374, -163.9949036, 163.9949036
17: -133.6509247, 71.8377533, -133.6509247, 71.8377533, -205.4886780, 205.4886780
18: -93.4816895, 69.9245148, -93.4816895, 69.9245148, -163.4062042, 163.4062042
19: -67.8313065, 40.4469528, -67.8313065, 40.4469528, -108.2782440, 108.2782516
20: -68.6554489, 53.1481323, -68.6554489, 53.1481323, -121.8035736, 121.8035736
21: -85.2860718, 51.2284088, -85.2860718, 51.2284088, -136.5144806, 136.5144806
22: -86.7702255, 46.5293999, -86.7702255, 46.5293999, -133.2996216, 133.2996216
23: -70.1346893, 54.1026115, -70.1346893, 54.1026115, -124.2373047, 124.2372971
24: -90.4928436, 54.5178070, -90.4928436, 54.5178070, -145.0106506, 145.0106506
25: -76.2291565, 55.5639420, -76.2291565, 55.5639420, -131.7930908, 131.7930908
26: -101.0771179, 82.1595459, -101.0771179, 82.1595459, -183.2366638, 183.2366638
27: -88.0060730, 49.6620827, -88.0060730, 49.6620827, -137.6681519, 137.6681519
28: -68.5764618, 54.5195351, -68.5764618, 54.5195351, -123.0959930, 123.0959854
29: -89.3036041, 41.8067436, -89.3036041, 41.8067436, -131.1103516, 131.1103516
30: -88.4475021, 63.7732086, -88.4475021, 63.7732086, -152.2207031, 152.2207031
31: -91.6715622, 56.0853958, -91.6715622, 56.0853958, -147.7569580, 147.7569580
32: -90.1399078, 57.5270233, -90.1399078, 57.5270233, -147.6669312, 147.6669312
33: -127.0187531, 78.1831512, -127.0187531, 78.1831512, -205.2019043, 205.2019043
34: -106.5280914, 48.8274612, -106.5280914, 48.8274612, -155.3555450, 155.3555450
35: -99.3480682, 58.9603500, -99.3480682, 58.9603500, -158.3084106, 158.3084106
36: -92.7557907, 57.3737221, -92.7557907, 57.3737221, -150.1294861, 150.1294861
37: -145.7819977, 62.9130173, -145.7819977, 62.9130173, -208.6950073, 208.6950073
38: -112.5812302, 71.4618912, -112.5812302, 71.4618912, -184.0431213, 184.0431213
39: -133.5139160, 76.7173309, -133.5139160, 76.7173309, -210.2312469, 210.2312469
40: -111.2424469, 56.9629440, -111.2424469, 56.9629440, -168.2053833, 168.2053833
41: -96.0516205, 65.9498138, -96.0516205, 65.9498138, -162.0014343, 162.0014343
42: -70.4349060, 56.7097397, -70.4349060, 56.7097397, -127.1446457, 127.1446457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=522, inp2_unstable=522, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 922
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 921
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 979
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 996
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 791
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1470
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1535
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 920
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 919
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1463
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 980
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1528
type: RSZ, layer: 1, pos: 957
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 811
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1447
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 790
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 981
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1462
type: RSZ, layer: 1, pos: 972
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 863
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 789
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1481
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1386
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 834
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1482
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1370
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1418

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 733

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1745

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -84.4781991, upper bound: 84.3624113
time: 108.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -84.5154984, upper bound: 84.3249996
time: 585.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 701.49 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 701.49
Output dim: 9, lower bound: -84.3382305, upper bound: 84.5013587
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 701.49
Output dim: 9, lower bound: -84.3755522, upper bound: 84.4639984
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 701.49
Output dim: 9, lower bound: -84.2849676, upper bound: 84.5541317
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 701.49
Output dim: 9, lower bound: -84.3222989, upper bound: 84.5167747
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 701.49
Output dim: 9, lower bound: -84.3041106, upper bound: 84.5352674
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 701.49
Output dim: 9, lower bound: -84.3413951, upper bound: 84.4978500
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 701.49
Output dim: 9, lower bound: -84.4592338, upper bound: 84.3815382
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 701.49
Output dim: 9, lower bound: -84.4965516, upper bound: 84.3441573
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 701.49
Output dim: 9, lower bound: -84.4781991, upper bound: 84.3624113
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 701.49
Output dim: 9, lower bound: -84.5154984, upper bound: 84.3249996
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 701.49
Output dim: 9, lower bound: -84.3652172, upper bound: 84.5177432
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 701.49
Output dim: 9, lower bound: -84.3843034, upper bound: 84.4988494
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 701.49
Output dim: 9, lower bound: -84.4988494, upper bound: 84.3843034
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 701.49
Output dim: 9, lower bound: -84.5177432, upper bound: 84.3652172
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 701.49
Output dim: 9, lower bound: -84.3649604, upper bound: 84.5180330
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 701.49
Output dim: 9, lower bound: -84.3840700, upper bound: 84.4991097
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 701.49
Output dim: 9, lower bound: -84.5377965, upper bound: 84.3439274
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 701.49
Output dim: 9, lower bound: -84.5566676, upper bound: 84.3248555
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 701.49
Output dim: 9, lower bound: -84.5038856, upper bound: 84.3781033

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 130.34 + 7108.45 = 7238.79 seconds
