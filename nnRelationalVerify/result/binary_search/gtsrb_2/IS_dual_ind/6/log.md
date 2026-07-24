## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 90.8802801588
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

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

## BASE Result
execution time: IAR + LP analysis = 2.90 + 122.79 = 125.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -95.6478369, upper bound: 95.6478369


# Binary Search by BASE starts (time budget: 17874.31 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=137.98736572265625
rel_dist={9: [-89.07095758814921, 89.07095758707153]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=137.98736572265625
rel_dist={9: [-92.6968285599128, 92.6968285599128]}

## Binary search (step 2) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=7, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=137.98736572265625
rel_dist={9: [-90.37261232965335, 90.37261232947226]}

## Binary search (step 3) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start
Binary search (step 3): status=Status.UNKNOWN, k_low=8, k_high=8, k_mid=8, eps_mid=0.0312500, abs_max=137.98736572265625
rel_dist={9: [-91.57507985954415, 91.57507985872158]}

## Binary Search Result
Binary search time: 745.84 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.02734375


# Individual Split (IS_dual_ind) starts
Time budget: 17128.47 seconds

## Binary search (step 0) starts
Candidate k: 10, corresponding eps: 0.0390625


## IAR start

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1685

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.7400021, upper bound: 93.5200792
time: 311.93 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.7400021, upper bound: 93.7400019
time: 94.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 406.87 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 406.87
Output dim: 9, lower bound: -93.7400021, upper bound: 93.5200792
IS_A2, status: Status.UNKNOWN, split count: 1, time: 406.87
Output dim: 9, lower bound: -93.7400021, upper bound: 93.7400019

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -108.6825104, 78.8020706, -108.9002228, 78.9408188, -187.6233215, 187.7022858
1: -57.0678215, 59.0297890, -57.1934052, 59.1218491, -116.1896667, 116.2231903
2: -49.3603210, 60.4166794, -49.4631004, 60.5102196, -109.8705215, 109.8797760
3: -62.0270157, 73.3680878, -62.1485748, 73.5530090, -135.5800171, 135.5166626
4: -64.6978760, 70.6075897, -64.8451996, 70.7645874, -135.4624634, 135.4527893
5: -59.6082916, 73.0030823, -59.7570419, 73.1902466, -132.7985382, 132.7601013
6: -94.2190552, 62.8569450, -94.4373322, 62.9708710, -157.1899109, 157.2942657
7: -66.5231934, 69.4289551, -66.7008286, 69.5784149, -136.1016083, 136.1297913
8: -81.3496399, 83.7510376, -81.4594803, 83.8903046, -165.2399292, 165.2105103
9: -60.6766510, 76.6171570, -60.9966087, 76.9369965, -137.6136475, 137.6137695
10: -88.3650818, 90.9195251, -88.8963318, 91.4252472, -179.7903137, 179.8158569
11: -85.7005463, 58.0707436, -86.0664978, 58.2503052, -143.9508514, 144.1372375
12: -97.5419464, 77.0192108, -97.7341919, 77.1488037, -174.6907349, 174.7534027
13: -84.9839172, 98.4149780, -85.1343384, 98.5777740, -183.5616913, 183.5493164
14: -144.1477051, 82.0576172, -144.5361176, 82.3987122, -226.5464172, 226.5937347
15: -78.6271057, 64.2450562, -78.7407684, 64.4023438, -143.0294495, 142.9858093
16: -90.9301682, 72.2346725, -91.3641281, 72.5685883, -163.4987488, 163.5988007
17: -133.2652130, 71.6172714, -133.6372375, 71.8215179, -205.0867310, 205.2545166
18: -93.2614899, 69.7886658, -93.4652786, 69.9121017, -163.1735840, 163.2539368
19: -67.6487885, 40.3830147, -67.8184052, 40.4411621, -108.0899506, 108.2014160
20: -68.4995728, 53.0521393, -68.6457214, 53.1390038, -121.6385803, 121.6978607
21: -85.0168381, 51.1076126, -85.2714462, 51.2162895, -136.2331238, 136.3790588
22: -86.4652023, 46.3651047, -86.7394485, 46.5208702, -132.9860687, 133.1045532
23: -69.9554596, 54.0113220, -70.1236038, 54.0935097, -124.0489655, 124.1349258
24: -90.2603989, 54.3981781, -90.4656372, 54.5113297, -144.7717285, 144.8638153
25: -76.0829163, 55.4306641, -76.2175217, 55.5554543, -131.6383667, 131.6481628
26: -100.8495712, 82.0273743, -101.0590591, 82.1481476, -182.9977112, 183.0864258
27: -87.7463531, 49.5539932, -87.9772949, 49.6556625, -137.4020081, 137.5312805
28: -68.4534607, 54.4348679, -68.5659637, 54.5132866, -122.9667511, 123.0008240
29: -89.0571289, 41.7203522, -89.2836990, 41.7999954, -130.8571167, 131.0040588
30: -88.2577209, 63.5894547, -88.4363708, 63.7538795, -152.0115814, 152.0258179
31: -91.4716110, 56.0199738, -91.6556931, 56.0788727, -147.5504761, 147.6756592
32: -89.9175873, 57.4322014, -90.1163483, 57.5225677, -147.4401550, 147.5485535
33: -126.6107407, 77.8046570, -126.9663086, 78.1773529, -204.7880859, 204.7709503
34: -106.3053589, 48.5926857, -106.5011139, 48.8209877, -155.1263428, 155.0937958
35: -99.0683594, 58.6328049, -99.3131180, 58.9556084, -158.0239563, 157.9459229
36: -92.4493256, 57.0919380, -92.7176590, 57.3697739, -149.8190918, 149.8096008
37: -145.2531738, 62.5798264, -145.7182617, 62.9070969, -208.1602783, 208.2980957
38: -112.1943970, 71.0787964, -112.5356369, 71.4542007, -183.6485901, 183.6144409
39: -133.0400085, 76.3493347, -133.4575195, 76.7122955, -209.7523041, 209.8068542
40: -110.8761749, 56.7141113, -111.2005463, 56.9587555, -167.8349304, 167.9146576
41: -95.7493744, 65.7828827, -96.0159149, 65.9457626, -161.6951294, 161.7987823
42: -70.2251740, 56.6016235, -70.4151154, 56.7022400, -126.9274139, 127.0167389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=522, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1685
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
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1579
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
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1462
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
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1528
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1685

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.5200792, upper bound: 93.5200792
time: 118.69 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.5200792, upper bound: 93.5200792
time: 135.22 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -108.9021606, 78.9381256, -108.9148483, 78.9571686, -187.8593292, 187.8529663
1: -57.1944237, 59.1250954, -57.1993790, 59.1304398, -116.3248596, 116.3244705
2: -49.4602089, 60.5124817, -49.4721794, 60.5177155, -109.9779205, 109.9846649
3: -62.1488037, 73.5440979, -62.1584053, 73.5738525, -135.7226562, 135.7024994
4: -64.8313599, 70.7698593, -64.8607330, 70.7776871, -135.6090393, 135.6305847
5: -59.7581177, 73.1889801, -59.7657928, 73.2125092, -132.9706268, 132.9547729
6: -94.4425659, 62.9719315, -94.4621735, 62.9764023, -157.4189758, 157.4340820
7: -66.7029877, 69.5845032, -66.7086792, 69.5975647, -136.3005524, 136.2931824
8: -81.4612885, 83.8939972, -81.4671555, 83.9014664, -165.3627472, 165.3611450
9: -60.9977417, 76.9633636, -61.0048904, 76.9823532, -137.9801025, 137.9682617
10: -88.8991547, 91.4643250, -88.9067078, 91.4981842, -180.3973389, 180.3710327
11: -86.0711594, 58.2591438, -86.0777893, 58.2742767, -144.3454285, 144.3369293
12: -97.7410355, 77.1502075, -97.7509918, 77.1578674, -174.8988953, 174.9011993
13: -85.1333160, 98.5814056, -85.1460342, 98.5895081, -183.7228241, 183.7274323
14: -144.5381165, 82.4331665, -144.5495911, 82.4509888, -226.9891052, 226.9827576
15: -78.7409515, 64.4085541, -78.7490082, 64.4167862, -143.1577454, 143.1575623
16: -91.3679581, 72.5947571, -91.3781128, 72.6166534, -163.9846191, 163.9728699
17: -133.6418152, 71.8285294, -133.6508789, 71.8377228, -205.4795380, 205.4794006
18: -93.4431305, 69.9135056, -93.4815140, 69.9244537, -163.3675842, 163.3950195
19: -67.8218231, 40.4408417, -67.8312836, 40.4469223, -108.2687225, 108.2721252
20: -68.6486969, 53.1403923, -68.6554108, 53.1481018, -121.7967834, 121.7957916
21: -85.2771988, 51.2163391, -85.2860031, 51.2283401, -136.5055237, 136.5023346
22: -86.7549820, 46.5233116, -86.7701492, 46.5293770, -133.2843475, 133.2934570
23: -70.1270905, 54.0933571, -70.1346588, 54.1025620, -124.2296524, 124.2280045
24: -90.4580688, 54.5093765, -90.4926758, 54.5177536, -144.9758148, 145.0020447
25: -76.2221527, 55.5569839, -76.2291107, 55.5639038, -131.7860565, 131.7861023
26: -101.0425262, 82.1481018, -101.0769653, 82.1594849, -183.2020111, 183.2250671
27: -87.9699097, 49.6536369, -88.0059052, 49.6620064, -137.6319122, 137.6595459
28: -68.5663452, 54.5140762, -68.5764084, 54.5195007, -123.0858459, 123.0904846
29: -89.2900162, 41.8007965, -89.3035202, 41.8067055, -131.0967102, 131.1043091
30: -88.4397125, 63.7614975, -88.4474640, 63.7731514, -152.2128601, 152.2089539
31: -91.6505814, 56.0764847, -91.6714706, 56.0853577, -147.7359314, 147.7479553
32: -90.1213760, 57.5232277, -90.1397858, 57.5269928, -147.6483612, 147.6630096
33: -126.9933014, 78.1782227, -127.0185852, 78.1831360, -205.1764374, 205.1967926
34: -106.5051193, 48.8233833, -106.5279388, 48.8274422, -155.3325500, 155.3513184
35: -99.3263855, 58.9572334, -99.3479309, 58.9603577, -158.2867432, 158.3051605
36: -92.7336502, 57.3715973, -92.7556686, 57.3736954, -150.1073303, 150.1272583
37: -145.7526855, 62.9080963, -145.7818298, 62.9129868, -208.6656799, 208.6899261
38: -112.5565109, 71.4561234, -112.5811157, 71.4618759, -184.0183716, 184.0372314
39: -133.4869385, 76.7122040, -133.5137939, 76.7173462, -210.2042847, 210.2259979
40: -111.2218399, 56.9596291, -111.2423248, 56.9629288, -168.1847687, 168.2019501
41: -96.0328751, 65.9454193, -96.0515137, 65.9497833, -161.9826355, 161.9969330
42: -70.4235611, 56.6825485, -70.4348526, 56.7096176, -127.1331558, 127.1174011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=522, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1685
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
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 857
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
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1626
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
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1036

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1685

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.5200792, upper bound: 93.7400021
time: 142.81 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.5200792, upper bound: 93.7400021
time: 106.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 251.37 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 251.37
Output dim: 9, lower bound: -93.5200792, upper bound: 93.5200792
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 251.37
Output dim: 9, lower bound: -93.5200792, upper bound: 93.5200792
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 251.37
Output dim: 9, lower bound: -93.5200792, upper bound: 93.7400021
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 251.37
Output dim: 9, lower bound: -93.5200792, upper bound: 93.7400021

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -108.6825104, 78.8020706, -108.6825104, 78.8020706, -187.4845734, 187.4845734
1: -57.0678215, 59.0297890, -57.0678215, 59.0297890, -116.0976105, 116.0976105
2: -49.3603210, 60.4166794, -49.3603210, 60.4166794, -109.7769928, 109.7770004
3: -62.0270157, 73.3680878, -62.0270157, 73.3680878, -135.3951111, 135.3950958
4: -64.6978760, 70.6075897, -64.6978760, 70.6075897, -135.3054504, 135.3054504
5: -59.6082916, 73.0030823, -59.6082916, 73.0030823, -132.6113739, 132.6113739
6: -94.2190552, 62.8569450, -94.2190552, 62.8569450, -157.0759888, 157.0759888
7: -66.5231934, 69.4289551, -66.5231934, 69.4289551, -135.9521484, 135.9521484
8: -81.3496399, 83.7510376, -81.3496399, 83.7510376, -165.1006775, 165.1006775
9: -60.6766510, 76.6171570, -60.6766510, 76.6171570, -137.2938080, 137.2938080
10: -88.3650818, 90.9195251, -88.3650818, 90.9195251, -179.2845917, 179.2845917
11: -85.7005463, 58.0707436, -85.7005463, 58.0707436, -143.7712860, 143.7712860
12: -97.5419464, 77.0192108, -97.5419464, 77.0192108, -174.5611420, 174.5611420
13: -84.9839172, 98.4149780, -84.9839172, 98.4149780, -183.3988953, 183.3988953
14: -144.1477051, 82.0576172, -144.1477051, 82.0576172, -226.2052917, 226.2053223
15: -78.6271057, 64.2450562, -78.6271057, 64.2450562, -142.8721619, 142.8721619
16: -90.9301682, 72.2346725, -90.9301682, 72.2346725, -163.1648407, 163.1648407
17: -133.2652130, 71.6172714, -133.2652130, 71.6172714, -204.8824768, 204.8824768
18: -93.2614899, 69.7886658, -93.2614899, 69.7886658, -163.0501404, 163.0501556
19: -67.6487885, 40.3830147, -67.6487885, 40.3830147, -108.0317993, 108.0317993
20: -68.4995728, 53.0521393, -68.4995728, 53.0521393, -121.5517120, 121.5517120
21: -85.0168381, 51.1076126, -85.0168381, 51.1076126, -136.1244507, 136.1244507
22: -86.4652023, 46.3651047, -86.4652023, 46.3651047, -132.8303070, 132.8303070
23: -69.9554596, 54.0113220, -69.9554596, 54.0113220, -123.9667816, 123.9667816
24: -90.2603989, 54.3981781, -90.2603989, 54.3981781, -144.6585693, 144.6585693
25: -76.0829163, 55.4306641, -76.0829163, 55.4306641, -131.5135803, 131.5135803
26: -100.8495712, 82.0273743, -100.8495712, 82.0273743, -182.8769379, 182.8769531
27: -87.7463531, 49.5539932, -87.7463531, 49.5539932, -137.3003540, 137.3003540
28: -68.4534607, 54.4348679, -68.4534607, 54.4348679, -122.8883133, 122.8883133
29: -89.0571289, 41.7203522, -89.0571289, 41.7203522, -130.7774811, 130.7774811
30: -88.2577209, 63.5894547, -88.2577209, 63.5894547, -151.8471680, 151.8471680
31: -91.4716110, 56.0199738, -91.4716110, 56.0199738, -147.4915771, 147.4915771
32: -89.9175873, 57.4322014, -89.9175873, 57.4322014, -147.3497772, 147.3497772
33: -126.6107407, 77.8046570, -126.6107407, 77.8046570, -204.4154053, 204.4153900
34: -106.3053589, 48.5926857, -106.3053589, 48.5926857, -154.8980255, 154.8980408
35: -99.0683594, 58.6328049, -99.0683594, 58.6328049, -157.7011566, 157.7011719
36: -92.4493256, 57.0919380, -92.4493256, 57.0919380, -149.5412598, 149.5412598
37: -145.2531738, 62.5798264, -145.2531738, 62.5798264, -207.8329773, 207.8330078
38: -112.1943970, 71.0787964, -112.1943970, 71.0787964, -183.2731934, 183.2731628
39: -133.0400085, 76.3493347, -133.0400085, 76.3493347, -209.3893280, 209.3893433
40: -110.8761749, 56.7141113, -110.8761749, 56.7141113, -167.5902863, 167.5902863
41: -95.7493744, 65.7828827, -95.7493744, 65.7828827, -161.5322571, 161.5322266
42: -70.2251740, 56.6016235, -70.2251740, 56.6016235, -126.8267975, 126.8267975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 856
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
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1464
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
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1738
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
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1669

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.5987338, upper bound: 93.4204840
time: 173.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.5987066, upper bound: 93.4998450
time: 92.45 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -108.6825104, 78.8020706, -108.9021606, 78.9381256, -187.6206360, 187.7042236
1: -57.0678215, 59.0297890, -57.1944237, 59.1250954, -116.1929169, 116.2242126
2: -49.3603210, 60.4166794, -49.4602089, 60.5124817, -109.8728027, 109.8768921
3: -62.0270157, 73.3680878, -62.1488037, 73.5440979, -135.5711060, 135.5168915
4: -64.6978760, 70.6075897, -64.8313599, 70.7698593, -135.4677124, 135.4389496
5: -59.6082916, 73.0030823, -59.7581177, 73.1889801, -132.7972717, 132.7612000
6: -94.2190552, 62.8569450, -94.4425659, 62.9719315, -157.1909790, 157.2994995
7: -66.5231934, 69.4289551, -66.7029877, 69.5845032, -136.1076965, 136.1319427
8: -81.3496399, 83.7510376, -81.4612885, 83.8939972, -165.2436218, 165.2123108
9: -60.6766510, 76.6171570, -60.9977417, 76.9633636, -137.6400146, 137.6148987
10: -88.3650818, 90.9195251, -88.8991547, 91.4643250, -179.8294067, 179.8186646
11: -85.7005463, 58.0707436, -86.0711594, 58.2591438, -143.9596863, 144.1419067
12: -97.5419464, 77.0192108, -97.7410355, 77.1502075, -174.6921539, 174.7602386
13: -84.9839172, 98.4149780, -85.1333160, 98.5814056, -183.5653229, 183.5482941
14: -144.1477051, 82.0576172, -144.5381165, 82.4331665, -226.5808716, 226.5957336
15: -78.6271057, 64.2450562, -78.7409515, 64.4085541, -143.0356598, 142.9860077
16: -90.9301682, 72.2346725, -91.3679581, 72.5947571, -163.5249176, 163.6026306
17: -133.2652130, 71.6172714, -133.6418152, 71.8285294, -205.0937500, 205.2590790
18: -93.2614899, 69.7886658, -93.4431305, 69.9135056, -163.1749725, 163.2317810
19: -67.6487885, 40.3830147, -67.8218231, 40.4408417, -108.0896301, 108.2048340
20: -68.4995728, 53.0521393, -68.6486969, 53.1403923, -121.6399536, 121.7008362
21: -85.0168381, 51.1076126, -85.2771988, 51.2163391, -136.2331696, 136.3848114
22: -86.4652023, 46.3651047, -86.7549820, 46.5233116, -132.9885101, 133.1200867
23: -69.9554596, 54.0113220, -70.1270905, 54.0933571, -124.0488129, 124.1384125
24: -90.2603989, 54.3981781, -90.4580688, 54.5093765, -144.7697754, 144.8562317
25: -76.0829163, 55.4306641, -76.2221527, 55.5569839, -131.6398926, 131.6528015
26: -100.8495712, 82.0273743, -101.0425262, 82.1481018, -182.9976807, 183.0699005
27: -87.7463531, 49.5539932, -87.9699097, 49.6536369, -137.3999939, 137.5238953
28: -68.4534607, 54.4348679, -68.5663452, 54.5140762, -122.9675293, 123.0012131
29: -89.0571289, 41.7203522, -89.2900162, 41.8007965, -130.8579254, 131.0103760
30: -88.2577209, 63.5894547, -88.4397125, 63.7614975, -152.0192108, 152.0291748
31: -91.4716110, 56.0199738, -91.6505814, 56.0764847, -147.5480957, 147.6705627
32: -89.9175873, 57.4322014, -90.1213760, 57.5232277, -147.4407959, 147.5535736
33: -126.6107407, 77.8046570, -126.9933014, 78.1782227, -204.7889404, 204.7979431
34: -106.3053589, 48.5926857, -106.5051193, 48.8233833, -155.1287231, 155.0977936
35: -99.0683594, 58.6328049, -99.3263855, 58.9572334, -158.0255890, 157.9591980
36: -92.4493256, 57.0919380, -92.7336502, 57.3715973, -149.8209229, 149.8255920
37: -145.2531738, 62.5798264, -145.7526855, 62.9080963, -208.1612549, 208.3325195
38: -112.1943970, 71.0787964, -112.5565109, 71.4561234, -183.6505127, 183.6352844
39: -133.0400085, 76.3493347, -133.4869385, 76.7122040, -209.7522125, 209.8362732
40: -110.8761749, 56.7141113, -111.2218399, 56.9596291, -167.8358002, 167.9359436
41: -95.7493744, 65.7828827, -96.0328751, 65.9454193, -161.6947937, 161.8157349
42: -70.2251740, 56.6016235, -70.4235611, 56.6825485, -126.9077225, 127.0251846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=521, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 856
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
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1464
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
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1738
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
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1669

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.5987338, upper bound: 93.4204840
time: 152.53 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.5987066, upper bound: 93.4998450
time: 128.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -108.9021606, 78.9381256, -108.6825104, 78.8020706, -187.7042084, 187.6206360
1: -57.1944237, 59.1250954, -57.0678215, 59.0297890, -116.2242126, 116.1929092
2: -49.4602089, 60.5124817, -49.3603210, 60.4166794, -109.8768921, 109.8728027
3: -62.1488037, 73.5440979, -62.0270157, 73.3680878, -135.5168915, 135.5711060
4: -64.8313599, 70.7698593, -64.6978760, 70.6075897, -135.4389496, 135.4677124
5: -59.7581177, 73.1889801, -59.6082916, 73.0030823, -132.7612000, 132.7972717
6: -94.4425659, 62.9719315, -94.2190552, 62.8569450, -157.2994995, 157.1909790
7: -66.7029877, 69.5845032, -66.5231934, 69.4289551, -136.1319427, 136.1076965
8: -81.4612885, 83.8939972, -81.3496399, 83.7510376, -165.2123260, 165.2436371
9: -60.9977417, 76.9633636, -60.6766510, 76.6171570, -137.6148987, 137.6400146
10: -88.8991547, 91.4643250, -88.3650818, 90.9195251, -179.8186798, 179.8294067
11: -86.0711594, 58.2591438, -85.7005463, 58.0707436, -144.1419067, 143.9596863
12: -97.7410355, 77.1502075, -97.5419464, 77.0192108, -174.7602386, 174.6921539
13: -85.1333160, 98.5814056, -84.9839172, 98.4149780, -183.5482788, 183.5653076
14: -144.5381165, 82.4331665, -144.1477051, 82.0576172, -226.5957336, 226.5808716
15: -78.7409515, 64.4085541, -78.6271057, 64.2450562, -142.9860077, 143.0356598
16: -91.3679581, 72.5947571, -90.9301682, 72.2346725, -163.6026306, 163.5249023
17: -133.6418152, 71.8285294, -133.2652130, 71.6172714, -205.2590942, 205.0937500
18: -93.4431305, 69.9135056, -93.2614899, 69.7886658, -163.2317810, 163.1749878
19: -67.8218231, 40.4408417, -67.6487885, 40.3830147, -108.2048340, 108.0896301
20: -68.6486969, 53.1403923, -68.4995728, 53.0521393, -121.7008209, 121.6399536
21: -85.2771988, 51.2163391, -85.0168381, 51.1076126, -136.3847961, 136.2331848
22: -86.7549820, 46.5233116, -86.4652023, 46.3651047, -133.1200867, 132.9885101
23: -70.1270905, 54.0933571, -69.9554596, 54.0113220, -124.1384125, 124.0488129
24: -90.4580688, 54.5093765, -90.2603989, 54.3981781, -144.8562469, 144.7697754
25: -76.2221527, 55.5569839, -76.0829163, 55.4306641, -131.6528015, 131.6398926
26: -101.0425262, 82.1481018, -100.8495712, 82.0273743, -183.0699005, 182.9976807
27: -87.9699097, 49.6536369, -87.7463531, 49.5539932, -137.5238953, 137.3999939
28: -68.5663452, 54.5140762, -68.4534607, 54.4348679, -123.0012131, 122.9675217
29: -89.2900162, 41.8007965, -89.0571289, 41.7203522, -131.0103760, 130.8579254
30: -88.4397125, 63.7614975, -88.2577209, 63.5894547, -152.0291748, 152.0192261
31: -91.6505814, 56.0764847, -91.4716110, 56.0199738, -147.6705475, 147.5480957
32: -90.1213760, 57.5232277, -89.9175873, 57.4322014, -147.5535736, 147.4407959
33: -126.9933014, 78.1782227, -126.6107407, 77.8046570, -204.7979431, 204.7889557
34: -106.5051193, 48.8233833, -106.3053589, 48.5926857, -155.0977936, 155.1287384
35: -99.3263855, 58.9572334, -99.0683594, 58.6328049, -157.9591980, 158.0255890
36: -92.7336502, 57.3715973, -92.4493256, 57.0919380, -149.8255920, 149.8209229
37: -145.7526855, 62.9080963, -145.2531738, 62.5798264, -208.3324890, 208.1612701
38: -112.5565109, 71.4561234, -112.1943970, 71.0787964, -183.6352692, 183.6505127
39: -133.4869385, 76.7122040, -133.0400085, 76.3493347, -209.8362732, 209.7522125
40: -111.2218399, 56.9596291, -110.8761749, 56.7141113, -167.9359436, 167.8358002
41: -96.0328751, 65.9454193, -95.7493744, 65.7828827, -161.8157349, 161.6947937
42: -70.4235611, 56.6825485, -70.2251740, 56.6016235, -127.0251846, 126.9077225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=521, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1669

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.4998451, upper bound: 93.5793751
time: 112.44 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.4998451, upper bound: 93.7271033
time: 122.41 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -108.9021606, 78.9381256, -108.9021606, 78.9381256, -187.8402710, 187.8402863
1: -57.1944237, 59.1250954, -57.1944237, 59.1250954, -116.3195190, 116.3195190
2: -49.4602089, 60.5124817, -49.4602089, 60.5124817, -109.9726868, 109.9726868
3: -62.1488037, 73.5440979, -62.1488037, 73.5440979, -135.6929016, 135.6929016
4: -64.8313599, 70.7698593, -64.8313599, 70.7698593, -135.6012268, 135.6012268
5: -59.7581177, 73.1889801, -59.7581177, 73.1889801, -132.9470978, 132.9470978
6: -94.4425659, 62.9719315, -94.4425659, 62.9719315, -157.4144897, 157.4144897
7: -66.7029877, 69.5845032, -66.7029877, 69.5845032, -136.2874908, 136.2874908
8: -81.4612885, 83.8939972, -81.4612885, 83.8939972, -165.3552551, 165.3552856
9: -60.9977417, 76.9633636, -60.9977417, 76.9633636, -137.9611053, 137.9611053
10: -88.8991547, 91.4643250, -88.8991547, 91.4643250, -180.3634796, 180.3634644
11: -86.0711594, 58.2591438, -86.0711594, 58.2591438, -144.3302917, 144.3303070
12: -97.7410355, 77.1502075, -97.7410355, 77.1502075, -174.8912354, 174.8912354
13: -85.1333160, 98.5814056, -85.1333160, 98.5814056, -183.7147064, 183.7147064
14: -144.5381165, 82.4331665, -144.5381165, 82.4331665, -226.9712830, 226.9712830
15: -78.7409515, 64.4085541, -78.7409515, 64.4085541, -143.1495056, 143.1495056
16: -91.3679581, 72.5947571, -91.3679581, 72.5947571, -163.9627075, 163.9627075
17: -133.6418152, 71.8285294, -133.6418152, 71.8285294, -205.4703369, 205.4703369
18: -93.4431305, 69.9135056, -93.4431305, 69.9135056, -163.3566284, 163.3566284
19: -67.8218231, 40.4408417, -67.8218231, 40.4408417, -108.2626648, 108.2626648
20: -68.6486969, 53.1403923, -68.6486969, 53.1403923, -121.7890930, 121.7890854
21: -85.2771988, 51.2163391, -85.2771988, 51.2163391, -136.4935303, 136.4935303
22: -86.7549820, 46.5233116, -86.7549820, 46.5233116, -133.2782898, 133.2782898
23: -70.1270905, 54.0933571, -70.1270905, 54.0933571, -124.2204361, 124.2204361
24: -90.4580688, 54.5093765, -90.4580688, 54.5093765, -144.9674377, 144.9674225
25: -76.2221527, 55.5569839, -76.2221527, 55.5569839, -131.7791443, 131.7791290
26: -101.0425262, 82.1481018, -101.0425262, 82.1481018, -183.1906128, 183.1906281
27: -87.9699097, 49.6536369, -87.9699097, 49.6536369, -137.6235504, 137.6235504
28: -68.5663452, 54.5140762, -68.5663452, 54.5140762, -123.0804214, 123.0804214
29: -89.2900162, 41.8007965, -89.2900162, 41.8007965, -131.0908203, 131.0908203
30: -88.4397125, 63.7614975, -88.4397125, 63.7614975, -152.2012024, 152.2012024
31: -91.6505814, 56.0764847, -91.6505814, 56.0764847, -147.7270660, 147.7270508
32: -90.1213760, 57.5232277, -90.1213760, 57.5232277, -147.6446075, 147.6446075
33: -126.9933014, 78.1782227, -126.9933014, 78.1782227, -205.1715088, 205.1715088
34: -106.5051193, 48.8233833, -106.5051193, 48.8233833, -155.3284912, 155.3285065
35: -99.3263855, 58.9572334, -99.3263855, 58.9572334, -158.2836151, 158.2836151
36: -92.7336502, 57.3715973, -92.7336502, 57.3715973, -150.1052399, 150.1052551
37: -145.7526855, 62.9080963, -145.7526855, 62.9080963, -208.6607666, 208.6607819
38: -112.5565109, 71.4561234, -112.5565109, 71.4561234, -184.0126038, 184.0126343
39: -133.4869385, 76.7122040, -133.4869385, 76.7122040, -210.1991272, 210.1991425
40: -111.2218399, 56.9596291, -111.2218399, 56.9596291, -168.1814728, 168.1814728
41: -96.0328751, 65.9454193, -96.0328751, 65.9454193, -161.9782715, 161.9782715
42: -70.4235611, 56.6825485, -70.4235611, 56.6825485, -127.1061096, 127.1061096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1669

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.4998451, upper bound: 93.5793754
time: 191.12 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.4998451, upper bound: 93.7271037
time: 154.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 347.98 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 347.98
Output dim: 9, lower bound: -93.5987338, upper bound: 93.4204840
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 347.98
Output dim: 9, lower bound: -93.5987066, upper bound: 93.4998450
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 347.98
Output dim: 9, lower bound: -93.5987338, upper bound: 93.4204840
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 347.98
Output dim: 9, lower bound: -93.5987066, upper bound: 93.4998450
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 347.98
Output dim: 9, lower bound: -93.4998451, upper bound: 93.5793751
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 347.98
Output dim: 9, lower bound: -93.4998451, upper bound: 93.7271033
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 347.98
Output dim: 9, lower bound: -93.4998451, upper bound: 93.5793754
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 347.98
Output dim: 9, lower bound: -93.4998451, upper bound: 93.7271037

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -108.5506516, 78.7246170, -108.6755600, 78.7980499, -187.3486938, 187.4001770
1: -56.9843712, 58.9458046, -57.0633926, 59.0260277, -116.0103989, 116.0092010
2: -49.2399521, 60.3107491, -49.3504143, 60.4126511, -109.6526031, 109.6611557
3: -61.8565826, 73.2091599, -62.0124321, 73.3634033, -135.2199707, 135.2215881
4: -64.5778809, 70.4876480, -64.6896057, 70.6027527, -135.1806335, 135.1772461
5: -59.4774971, 72.8726349, -59.5985718, 72.9985123, -132.4760132, 132.4712067
6: -93.9823608, 62.7458153, -94.1996613, 62.8538437, -156.8362122, 156.9454651
7: -66.3727417, 69.3187027, -66.5130005, 69.4249420, -135.7976685, 135.8316956
8: -81.2441406, 83.6334381, -81.3429794, 83.7462845, -164.9904175, 164.9764099
9: -60.4303360, 76.3844223, -60.6716766, 76.5944672, -137.0247955, 137.0560913
10: -87.8765335, 90.4492493, -88.3586960, 90.8729935, -178.7495270, 178.8079376
11: -85.4578476, 57.9242706, -85.6950531, 58.0576134, -143.5154572, 143.6193237
12: -97.2437210, 76.7960815, -97.5354767, 76.9986649, -174.2423553, 174.3315582
13: -84.8233566, 98.2853851, -84.9761200, 98.4076385, -183.2309570, 183.2615051
14: -143.7672119, 81.7253189, -144.1388855, 82.0244598, -225.7916718, 225.8641968
15: -78.4989471, 64.0957718, -78.6205215, 64.2351837, -142.7341309, 142.7162781
16: -90.6548080, 72.0003891, -90.9223099, 72.2133179, -162.8681335, 162.9226990
17: -133.0215454, 71.4710007, -133.2593384, 71.6067352, -204.6282806, 204.7303467
18: -93.1057663, 69.6566620, -93.2544556, 69.7806473, -162.8864136, 162.9111176
19: -67.4990768, 40.2989426, -67.6423798, 40.3772278, -107.8762970, 107.9413223
20: -68.3830109, 52.9986687, -68.4939575, 53.0495758, -121.4325867, 121.4926071
21: -84.8121872, 50.9863510, -85.0096512, 51.0977936, -135.9099731, 135.9960022
22: -86.3031616, 46.2641792, -86.4551010, 46.3587227, -132.6618805, 132.7192841
23: -69.8331451, 53.9347229, -69.9503632, 54.0062943, -123.8394394, 123.8850861
24: -90.1385345, 54.3005524, -90.2513809, 54.3948669, -144.5333862, 144.5519257
25: -75.9751282, 55.3318787, -76.0775375, 55.4256516, -131.4007568, 131.4094238
26: -100.6410675, 81.8202820, -100.8427429, 82.0101471, -182.6511993, 182.6630249
27: -87.5329666, 49.4462128, -87.7285843, 49.5509949, -137.0839539, 137.1748047
28: -68.3119354, 54.3198624, -68.4424362, 54.4313316, -122.7432709, 122.7622986
29: -88.9080734, 41.6175652, -89.0485992, 41.7116013, -130.6196747, 130.6661682
30: -88.1417923, 63.4941139, -88.2515869, 63.5832520, -151.7250366, 151.7456970
31: -91.3166199, 55.9384460, -91.4639282, 56.0144997, -147.3311157, 147.4023743
32: -89.7597885, 57.3673592, -89.9062424, 57.4298782, -147.1896667, 147.2736053
33: -126.3591843, 77.5131989, -126.5866241, 77.8001404, -204.1593323, 204.0998230
34: -106.0513229, 48.3582001, -106.2815628, 48.5884933, -154.6398163, 154.6397552
35: -98.8432693, 58.3575058, -99.0466919, 58.6295547, -157.4728241, 157.4041748
36: -92.2039261, 56.9101715, -92.4274750, 57.0898285, -149.2937622, 149.3376465
37: -145.0395966, 62.4041977, -145.2372284, 62.5751457, -207.6147461, 207.6414185
38: -111.8955765, 70.8134384, -112.1676483, 71.0741043, -182.9696808, 182.9810791
39: -132.8282166, 76.1567230, -133.0242920, 76.3456726, -209.1738892, 209.1810150
40: -110.6603088, 56.5544662, -110.8582535, 56.7118454, -167.3721619, 167.4127197
41: -95.5523834, 65.6622238, -95.7330856, 65.7796783, -161.3320618, 161.3953094
42: -70.0766296, 56.5032234, -70.2159424, 56.5968475, -126.6734467, 126.7191620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1669
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
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1699
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
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 753
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
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1689
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
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 956
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
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 934
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
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 851
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
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1786
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
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 537
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
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1649
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.3455116, upper bound: 93.4749919
time: 222.21 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.3455115, upper bound: 93.4749919
time: 143.75 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -108.6696320, 78.7972336, -108.6815948, 78.8017120, -187.4713440, 187.4788208
1: -57.0531540, 59.0252686, -57.0668106, 59.0294685, -116.0826187, 116.0920792
2: -49.3495026, 60.4134102, -49.3595734, 60.4164314, -109.7659302, 109.7729721
3: -62.0051155, 73.3640747, -62.0253410, 73.3677902, -135.3728943, 135.3894043
4: -64.6748352, 70.6014099, -64.6963120, 70.6071014, -135.2819214, 135.2977142
5: -59.5821762, 72.9992371, -59.6063423, 73.0028381, -132.5850220, 132.6055756
6: -94.2035522, 62.8539047, -94.2179413, 62.8567162, -157.0602722, 157.0718384
7: -66.5003357, 69.4254608, -66.5216217, 69.4287033, -135.9290466, 135.9470673
8: -81.3386002, 83.7470093, -81.3487701, 83.7507172, -165.0893250, 165.0957642
9: -60.6712837, 76.6020432, -60.6762619, 76.6160812, -137.2873535, 137.2782898
10: -88.3595886, 90.8887634, -88.3647079, 90.9173737, -179.2769623, 179.2534790
11: -85.6944580, 58.0569115, -85.7000580, 58.0698051, -143.7642670, 143.7569580
12: -97.5363922, 76.9938126, -97.5415192, 77.0171204, -174.5535126, 174.5353394
13: -84.9743195, 98.4069366, -84.9832077, 98.4144135, -183.3887329, 183.3901367
14: -144.1387329, 82.0367584, -144.1470795, 82.0559311, -226.1946716, 226.1838379
15: -78.6189194, 64.2293472, -78.6265259, 64.2438812, -142.8627930, 142.8558655
16: -90.9219894, 72.2189255, -90.9295883, 72.2335510, -163.1555328, 163.1485138
17: -133.2601013, 71.5994720, -133.2648163, 71.6160889, -204.8761902, 204.8642883
18: -93.2502136, 69.7798157, -93.2606735, 69.7880325, -163.0382385, 163.0404968
19: -67.6431503, 40.3744545, -67.6483307, 40.3824005, -108.0255432, 108.0227814
20: -68.4927826, 53.0498352, -68.4990616, 53.0519867, -121.5447693, 121.5488892
21: -85.0095367, 51.0976562, -85.0162659, 51.1069221, -136.1164551, 136.1139221
22: -86.4517441, 46.3523750, -86.4642029, 46.3641624, -132.8159027, 132.8165741
23: -69.9499817, 54.0042496, -69.9550476, 54.0108337, -123.9608078, 123.9592972
24: -90.2474670, 54.3928604, -90.2594986, 54.3978004, -144.6452637, 144.6523590
25: -76.0774384, 55.4253044, -76.0824814, 55.4302750, -131.5077209, 131.5077820
26: -100.8419876, 82.0008087, -100.8489761, 82.0255585, -182.8675537, 182.8497772
27: -87.7305756, 49.5499191, -87.7452011, 49.5536919, -137.2842712, 137.2951202
28: -68.4424133, 54.4309959, -68.4526062, 54.4345779, -122.8769913, 122.8835907
29: -89.0498962, 41.7086792, -89.0565491, 41.7193718, -130.7692719, 130.7652283
30: -88.2500000, 63.5830078, -88.2570953, 63.5889893, -151.8389893, 151.8400879
31: -91.4648285, 56.0099602, -91.4711075, 56.0192337, -147.4840698, 147.4810638
32: -89.9080353, 57.4292450, -89.9168701, 57.4319878, -147.3400269, 147.3461151
33: -126.5937729, 77.7999420, -126.6095352, 77.8043365, -204.3981018, 204.4094849
34: -106.2891083, 48.5881233, -106.3042145, 48.5923691, -154.8814697, 154.8923340
35: -99.0533829, 58.6301880, -99.0673065, 58.6325760, -157.6859589, 157.6974945
36: -92.4339600, 57.0902100, -92.4482193, 57.0918121, -149.5257721, 149.5384216
37: -145.2404938, 62.5749664, -145.2522583, 62.5794487, -207.8199463, 207.8272095
38: -112.1757736, 71.0741577, -112.1931076, 71.0784607, -183.2542114, 183.2672424
39: -133.0268860, 76.3446198, -133.0390472, 76.3489914, -209.3758850, 209.3836670
40: -110.8625259, 56.7115517, -110.8752441, 56.7139206, -167.5764313, 167.5867920
41: -95.7368851, 65.7791443, -95.7484741, 65.7825851, -161.5194702, 161.5276184
42: -70.2154694, 56.5964203, -70.2244415, 56.6012268, -126.8166962, 126.8208618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1669
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
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1699
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
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1731
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
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 876
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
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 545
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
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1689
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
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 956
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
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 934
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
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 851
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
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1786
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
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 537
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
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 984
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
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1649
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.3455104, upper bound: 93.5904666
time: 102.31 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.3455104, upper bound: 93.5904666
time: 151.37 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -108.5506516, 78.7246170, -108.8954163, 78.9340286, -187.4846497, 187.6200256
1: -56.9843712, 58.9458046, -57.1901894, 59.1213303, -116.1056976, 116.1359940
2: -49.2399521, 60.3107491, -49.4502296, 60.5084496, -109.7483978, 109.7609711
3: -61.8565826, 73.2091599, -62.1343651, 73.5392761, -135.3958588, 135.3435059
4: -64.5778809, 70.4876480, -64.8228073, 70.7650299, -135.3429108, 135.3104401
5: -59.4774971, 72.8726349, -59.7484589, 73.1842728, -132.6617737, 132.6210938
6: -93.9823608, 62.7458153, -94.4231567, 62.9688034, -156.9511719, 157.1689606
7: -66.3727417, 69.3187027, -66.6933746, 69.5803528, -135.9530945, 136.0120850
8: -81.2441406, 83.6334381, -81.4547272, 83.8892975, -165.1334381, 165.0881653
9: -60.4303360, 76.3844223, -60.9927444, 76.9407654, -137.3710938, 137.3771667
10: -87.8765335, 90.4492493, -88.8927841, 91.4179001, -179.2944336, 179.3420258
11: -85.4578476, 57.9242706, -86.0657578, 58.2459106, -143.7037659, 143.9900208
12: -97.2437210, 76.7960815, -97.7345047, 77.1298676, -174.3735352, 174.5305786
13: -84.8233566, 98.2853851, -85.1256790, 98.5740051, -183.3973389, 183.4110413
14: -143.7672119, 81.7253189, -144.5293884, 82.3988800, -226.1660919, 226.2546844
15: -78.4989471, 64.0957718, -78.7343140, 64.3987122, -142.8976440, 142.8300781
16: -90.6548080, 72.0003891, -91.3601379, 72.5734863, -163.2283020, 163.3605347
17: -133.0215454, 71.4710007, -133.6361084, 71.8180084, -204.8395538, 205.1071167
18: -93.1057663, 69.6566620, -93.4361343, 69.9055328, -163.0112915, 163.0927887
19: -67.4990768, 40.2989426, -67.8153687, 40.4350891, -107.9341507, 108.1143112
20: -68.3830109, 52.9986687, -68.6430969, 53.1378250, -121.5208359, 121.6417694
21: -84.8121872, 50.9863510, -85.2700043, 51.2065277, -136.0187073, 136.2563477
22: -86.3031616, 46.2641792, -86.7447205, 46.5170670, -132.8202209, 133.0088959
23: -69.8331451, 53.9347229, -70.1219635, 54.0882034, -123.9213486, 124.0566864
24: -90.1385345, 54.3005524, -90.4485931, 54.5060501, -144.6445770, 144.7491455
25: -75.9751282, 55.3318787, -76.2168427, 55.5520287, -131.5271606, 131.5487213
26: -100.6410675, 81.8202820, -101.0356750, 82.1310043, -182.7720642, 182.8559570
27: -87.5329666, 49.4462128, -87.9514084, 49.6506691, -137.1836395, 137.3976135
28: -68.3119354, 54.3198624, -68.5552368, 54.5105438, -122.8224792, 122.8750992
29: -88.9080734, 41.6175652, -89.2813797, 41.7922363, -130.7003174, 130.8989410
30: -88.1417923, 63.4941139, -88.4336777, 63.7552414, -151.8970337, 151.9277954
31: -91.3166199, 55.9384460, -91.6429138, 56.0709648, -147.3875732, 147.5813599
32: -89.7597885, 57.3673592, -90.1098709, 57.5209427, -147.2807312, 147.4772339
33: -126.3591843, 77.5131989, -126.9691162, 78.1736755, -204.5328674, 204.4822998
34: -106.0513229, 48.3582001, -106.4813385, 48.8192596, -154.8705750, 154.8395386
35: -98.8432693, 58.3575058, -99.3042145, 58.9540329, -157.7972870, 157.6617126
36: -92.2039261, 56.9101715, -92.7118454, 57.3694992, -149.5734253, 149.6220093
37: -145.0395966, 62.4041977, -145.7367249, 62.9035454, -207.9431305, 208.1409302
38: -111.8955765, 70.8134384, -112.5299759, 71.4515305, -183.3471069, 183.3434143
39: -132.8282166, 76.1567230, -133.4711456, 76.7085342, -209.5367432, 209.6278687
40: -110.6603088, 56.5544662, -111.2038116, 56.9573898, -167.6176910, 167.7582703
41: -95.5523834, 65.6622238, -96.0164108, 65.9422684, -161.4946289, 161.6786194
42: -70.0766296, 56.5032234, -70.4143982, 56.6778221, -126.7544479, 126.9176102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1669
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
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 878
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
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1528
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.3455116, upper bound: 93.4122011
time: 119.66 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.3455116, upper bound: 93.4122011
time: 119.68 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -108.6696320, 78.7972336, -108.9011841, 78.9377747, -187.6074066, 187.6984253
1: -57.0531540, 59.0252686, -57.1933060, 59.1247749, -116.1779175, 116.2185745
2: -49.3495026, 60.4134102, -49.4595490, 60.5122604, -109.8617630, 109.8729553
3: -62.0051155, 73.3640747, -62.1471176, 73.5438232, -135.5489349, 135.5112000
4: -64.6748352, 70.6014099, -64.8299484, 70.7694244, -135.4442596, 135.4313660
5: -59.5821762, 72.9992371, -59.7561150, 73.1887207, -132.7708740, 132.7553558
6: -94.2035522, 62.8539047, -94.4414520, 62.9716606, -157.1752014, 157.2953491
7: -66.5003357, 69.4254608, -66.7011185, 69.5842590, -136.0845947, 136.1265869
8: -81.3386002, 83.7470093, -81.4603729, 83.8936615, -165.2322540, 165.2073669
9: -60.6712837, 76.6020432, -60.9973526, 76.9622955, -137.6335754, 137.5993958
10: -88.3595886, 90.8887634, -88.8987427, 91.4621277, -179.8217163, 179.7875061
11: -85.6944580, 58.0569115, -86.0706863, 58.2583046, -143.9527588, 144.1275940
12: -97.5363922, 76.9938126, -97.7406311, 77.1480713, -174.6844635, 174.7344360
13: -84.9743195, 98.4069366, -85.1325989, 98.5808029, -183.5551147, 183.5395355
14: -144.1387329, 82.0367584, -144.5374451, 82.4321060, -226.5708313, 226.5742035
15: -78.6189194, 64.2293472, -78.7403717, 64.4074097, -143.0263062, 142.9697266
16: -90.9219894, 72.2189255, -91.3673325, 72.5935822, -163.5155640, 163.5862427
17: -133.2601013, 71.5994720, -133.6414795, 71.8273163, -205.0874176, 205.2409363
18: -93.2502136, 69.7798157, -93.4422913, 69.9128571, -163.1630707, 163.2221069
19: -67.6431503, 40.3744545, -67.8214035, 40.4401855, -108.0833359, 108.1958466
20: -68.4927826, 53.0498352, -68.6481781, 53.1402054, -121.6329880, 121.6980133
21: -85.0095367, 51.0976562, -85.2766266, 51.2156029, -136.2251434, 136.3742828
22: -86.4517441, 46.3523750, -86.7540359, 46.5222664, -132.9739990, 133.1064148
23: -69.9499817, 54.0042496, -70.1266556, 54.0928612, -124.0428467, 124.1308975
24: -90.2474670, 54.3928604, -90.4573441, 54.5089684, -144.7564392, 144.8501892
25: -76.0774384, 55.4253044, -76.2217407, 55.5565796, -131.6340179, 131.6470490
26: -100.8419876, 82.0008087, -101.0419235, 82.1462250, -182.9882202, 183.0427246
27: -87.7305756, 49.5499191, -87.9689865, 49.6533585, -137.3839264, 137.5189056
28: -68.4424133, 54.4309959, -68.5655289, 54.5137978, -122.9561996, 122.9965134
29: -89.0498962, 41.7086792, -89.2894821, 41.7997704, -130.8496552, 130.9981689
30: -88.2500000, 63.5830078, -88.4390564, 63.7610626, -152.0110626, 152.0220642
31: -91.4648285, 56.0099602, -91.6501083, 56.0757179, -147.5405426, 147.6600647
32: -89.9080353, 57.4292450, -90.1207886, 57.5229607, -147.4309998, 147.5500336
33: -126.5937729, 77.7999420, -126.9921265, 78.1778564, -204.7716370, 204.7920532
34: -106.2891083, 48.5881233, -106.5039673, 48.8230743, -155.1121826, 155.0920715
35: -99.0533829, 58.6301880, -99.3255768, 58.9570541, -158.0104218, 157.9557648
36: -92.4339600, 57.0902100, -92.7325745, 57.3714409, -149.8054047, 149.8227844
37: -145.2404938, 62.5749664, -145.7518311, 62.9077530, -208.1482391, 208.3267975
38: -112.1757736, 71.0741577, -112.5550995, 71.4557800, -183.6315613, 183.6292419
39: -133.0268860, 76.3446198, -133.4860535, 76.7118607, -209.7387390, 209.8306580
40: -110.8625259, 56.7115517, -111.2209244, 56.9594460, -167.8219299, 167.9324646
41: -95.7368851, 65.7791443, -96.0320740, 65.9451141, -161.6820068, 161.8112183
42: -70.2154694, 56.5964203, -70.4228668, 56.6821289, -126.8975983, 127.0192871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1669
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.3455104, upper bound: 93.4916390
time: 137.46 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.3455104, upper bound: 93.4916390
time: 118.10 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -108.7691269, 78.8597031, -108.6755600, 78.7980499, -187.5671692, 187.5352631
1: -57.1112404, 59.0399437, -57.0633926, 59.0260277, -116.1372681, 116.1033325
2: -49.3381119, 60.4025497, -49.3504143, 60.4126511, -109.7507629, 109.7529602
3: -61.9776039, 73.3867645, -62.0124321, 73.3634033, -135.3410034, 135.3991852
4: -64.7092743, 70.6392441, -64.6896057, 70.6027527, -135.3120270, 135.3288574
5: -59.6292381, 73.0639496, -59.5985718, 72.9985123, -132.6277466, 132.6625214
6: -94.2014771, 62.8601379, -94.1996613, 62.8538437, -157.0553284, 157.0597839
7: -66.5531235, 69.4802170, -66.5130005, 69.4249420, -135.9780579, 135.9932251
8: -81.3548126, 83.7747879, -81.3429794, 83.7462845, -165.1010742, 165.1177673
9: -60.7492371, 76.7307205, -60.6716766, 76.5944672, -137.3437042, 137.4024048
10: -88.4010849, 90.9941559, -88.3586960, 90.8729935, -179.2740784, 179.3528442
11: -85.8240891, 58.1127167, -85.6950531, 58.0576134, -143.8816986, 143.8077545
12: -97.4342194, 76.9273605, -97.5354767, 76.9986649, -174.4328766, 174.4628296
13: -84.9687271, 98.4491196, -84.9761200, 98.4076385, -183.3763733, 183.4252319
14: -144.1347656, 82.0886383, -144.1388855, 82.0244598, -226.1592102, 226.2275238
15: -78.6115494, 64.2577362, -78.6205215, 64.2351837, -142.8467255, 142.8782349
16: -91.0903397, 72.3601761, -90.9223099, 72.2133179, -163.3036499, 163.2824707
17: -133.3899841, 71.6802597, -133.2593384, 71.6067352, -204.9967194, 204.9396057
18: -93.2854156, 69.7804337, -93.2544556, 69.7806473, -163.0660400, 163.0348816
19: -67.6704559, 40.3561172, -67.6423798, 40.3772278, -108.0476837, 107.9984894
20: -68.5305328, 53.0863914, -68.4939575, 53.0495758, -121.5801086, 121.5803299
21: -85.0669785, 51.0934486, -85.0096512, 51.0977936, -136.1647644, 136.1030884
22: -86.5905533, 46.4214401, -86.4551010, 46.3587227, -132.9492645, 132.8765411
23: -70.0036850, 54.0150833, -69.9503632, 54.0062943, -124.0099716, 123.9654388
24: -90.3324509, 54.4058647, -90.2513809, 54.3948669, -144.7272949, 144.6572266
25: -76.1137390, 55.4565430, -76.0775375, 55.4256516, -131.5393982, 131.5340881
26: -100.8341446, 81.9402466, -100.8427429, 82.0101471, -182.8442841, 182.7829895
27: -87.7478027, 49.5305710, -87.7285843, 49.5509949, -137.2987976, 137.2591553
28: -68.4231949, 54.3958473, -68.4424362, 54.4313316, -122.8545227, 122.8382797
29: -89.1385880, 41.6981583, -89.0485992, 41.7116013, -130.8501892, 130.7467651
30: -88.3236847, 63.6664124, -88.2515869, 63.5832520, -151.9069366, 151.9179993
31: -91.4931259, 55.9940186, -91.4639282, 56.0144997, -147.5076294, 147.4579468
32: -89.9597626, 57.4589539, -89.9062424, 57.4298782, -147.3896332, 147.3652039
33: -126.7408295, 77.8821869, -126.5866241, 77.8001404, -204.5409698, 204.4687958
34: -106.2507324, 48.5847855, -106.2815628, 48.5884933, -154.8392334, 154.8663330
35: -99.0961761, 58.6760941, -99.0466919, 58.6295547, -157.7257385, 157.7227631
36: -92.4875031, 57.1874619, -92.4274750, 57.0898285, -149.5773315, 149.6149292
37: -145.5373383, 62.7311172, -145.2372284, 62.5751457, -208.1124573, 207.9683533
38: -112.2578125, 71.1898727, -112.1676483, 71.0741043, -183.3319092, 183.3575134
39: -133.2735596, 76.5181274, -133.0242920, 76.3456726, -209.6192322, 209.5424194
40: -111.0020142, 56.7992134, -110.8582535, 56.7118454, -167.7138672, 167.6574707
41: -95.8308868, 65.8236084, -95.7330856, 65.7796783, -161.6105652, 161.5567017
42: -70.2700653, 56.5845528, -70.2159424, 56.5968475, -126.8668976, 126.8004913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1669
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
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1699
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
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 753
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
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1689
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
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 956
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
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 934
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
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 851
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
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1786
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
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 537
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
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1649
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.2472191, upper bound: 93.5711276
time: 113.80 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.2472191, upper bound: 93.5711276
time: 104.72 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -108.8881760, 78.9333801, -108.6815948, 78.8017120, -187.6898804, 187.6149750
1: -57.1783676, 59.1207809, -57.0668106, 59.0294685, -116.2078400, 116.1875916
2: -49.4512100, 60.5093079, -49.3595734, 60.4164314, -109.8676376, 109.8688812
3: -62.1267357, 73.5401764, -62.0253410, 73.3677902, -135.4945221, 135.5655212
4: -64.8107834, 70.7641525, -64.6963120, 70.6071014, -135.4178772, 135.4604645
5: -59.7317963, 73.1853485, -59.6063423, 73.0028381, -132.7346344, 132.7916870
6: -94.4270782, 62.9688721, -94.2179413, 62.8567162, -157.2837982, 157.1868134
7: -66.6768188, 69.5813904, -66.5216217, 69.4287033, -136.1055298, 136.1030121
8: -81.4500580, 83.8899536, -81.3487701, 83.7507172, -165.2007751, 165.2387238
9: -60.9925766, 76.9482880, -60.6762619, 76.6160812, -137.6086426, 137.6245422
10: -88.8938599, 91.4351959, -88.3647079, 90.9173737, -179.8112030, 179.7998810
11: -86.0649338, 58.2475739, -85.7000580, 58.0698051, -144.1347351, 143.9476318
12: -97.7356873, 77.1202393, -97.5415192, 77.0171204, -174.7527924, 174.6617584
13: -85.1236420, 98.5734406, -84.9832077, 98.4144135, -183.5380249, 183.5566406
14: -144.5291443, 82.4179993, -144.1470795, 82.0559311, -226.5850372, 226.5650787
15: -78.7333527, 64.3931198, -78.6265259, 64.2438812, -142.9772339, 143.0196533
16: -91.3596878, 72.5791168, -90.9295883, 72.2335510, -163.5932312, 163.5086975
17: -133.6365967, 71.8112030, -133.2648163, 71.6160889, -205.2526703, 205.0760193
18: -93.4316483, 69.9046707, -93.2606735, 69.7880325, -163.2196808, 163.1653442
19: -67.8163757, 40.4319000, -67.6483307, 40.3824005, -108.1987762, 108.0802307
20: -68.6418991, 53.1381073, -68.4990616, 53.0519867, -121.6938858, 121.6371613
21: -85.2698364, 51.2061234, -85.0162659, 51.1069221, -136.3767548, 136.2223816
22: -86.7420425, 46.5094986, -86.4642029, 46.3641624, -133.1062012, 132.9736938
23: -70.1215515, 54.0864906, -69.9550476, 54.0108337, -124.1323776, 124.0415344
24: -90.4482193, 54.5041351, -90.2594986, 54.3978004, -144.8460083, 144.7636414
25: -76.2166595, 55.5516205, -76.0824814, 55.4302750, -131.6469421, 131.6340942
26: -101.0347443, 82.1208878, -100.8489761, 82.0255585, -183.0603027, 182.9698639
27: -87.9572906, 49.6496468, -87.7452011, 49.5536919, -137.5109863, 137.3948364
28: -68.5550995, 54.5102081, -68.4526062, 54.4345779, -122.9896774, 122.9628067
29: -89.2834702, 41.7883377, -89.0565491, 41.7193718, -131.0028381, 130.8448792
30: -88.4310837, 63.7554970, -88.2570953, 63.5889893, -152.0200806, 152.0125732
31: -91.6441650, 56.0662422, -91.4711075, 56.0192337, -147.6633911, 147.5373535
32: -90.1130066, 57.5197601, -89.9168701, 57.4319878, -147.5449829, 147.4366302
33: -126.9794235, 78.1736298, -126.6095352, 77.8043365, -204.7837372, 204.7831726
34: -106.4887695, 48.8188782, -106.3042145, 48.5923691, -155.0811462, 155.1230927
35: -99.3154907, 58.9547234, -99.0673065, 58.6325760, -157.9480591, 158.0220337
36: -92.7207184, 57.3700104, -92.4482193, 57.0918121, -149.8125305, 149.8182373
37: -145.7403259, 62.9034195, -145.2522583, 62.5794487, -208.3197784, 208.1556702
38: -112.5373230, 71.4513550, -112.1931076, 71.0784607, -183.6157837, 183.6444397
39: -133.4744263, 76.7074738, -133.0390472, 76.3489914, -209.8234100, 209.7465210
40: -111.2088013, 56.9570427, -110.8752441, 56.7139206, -167.9227295, 167.8322906
41: -96.0227432, 65.9413910, -95.7484741, 65.7825851, -161.8053284, 161.6898651
42: -70.4143677, 56.6772156, -70.2244415, 56.6012268, -127.0155945, 126.9016495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1669
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
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1699
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
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1731
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
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 876
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
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 545
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
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1689
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
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 956
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
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 934
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
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 851
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
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1786
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
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 537
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
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 984
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
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1033
type: B, layer: 1, pos: 1027
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1649
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.2472191, upper bound: 93.7189853
time: 104.55 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.2472191, upper bound: 93.7189853
time: 138.54 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -108.7691269, 78.8597031, -108.8954163, 78.9340286, -187.7031555, 187.7550964
1: -57.1112404, 59.0399437, -57.1901894, 59.1213303, -116.2325745, 116.2301254
2: -49.3381119, 60.4025497, -49.4502296, 60.5084496, -109.8465576, 109.8527756
3: -61.9776039, 73.3867645, -62.1343651, 73.5392761, -135.5168762, 135.5211182
4: -64.7092743, 70.6392441, -64.8228073, 70.7650299, -135.4743042, 135.4620514
5: -59.6292381, 73.0639496, -59.7484589, 73.1842728, -132.8135071, 132.8124084
6: -94.2014771, 62.8601379, -94.4231567, 62.9688034, -157.1702881, 157.2832794
7: -66.5531235, 69.4802170, -66.6933746, 69.5803528, -136.1334534, 136.1735840
8: -81.3548126, 83.7747879, -81.4547272, 83.8892975, -165.2441101, 165.2295227
9: -60.7492371, 76.7307205, -60.9927444, 76.9407654, -137.6900024, 137.7234650
10: -88.4010849, 90.9941559, -88.8927841, 91.4179001, -179.8189850, 179.8869324
11: -85.8240891, 58.1127167, -86.0657578, 58.2459106, -144.0700073, 144.1784668
12: -97.4342194, 76.9273605, -97.7345047, 77.1298676, -174.5640869, 174.6618652
13: -84.9687271, 98.4491196, -85.1256790, 98.5740051, -183.5427246, 183.5747986
14: -144.1347656, 82.0886383, -144.5293884, 82.3988800, -226.5336456, 226.6180267
15: -78.6115494, 64.2577362, -78.7343140, 64.3987122, -143.0102539, 142.9920349
16: -91.0903397, 72.3601761, -91.3601379, 72.5734863, -163.6638184, 163.7203064
17: -133.3899841, 71.6802597, -133.6361084, 71.8180084, -205.2079926, 205.3163757
18: -93.2854156, 69.7804337, -93.4361343, 69.9055328, -163.1909180, 163.2165680
19: -67.6704559, 40.3561172, -67.8153687, 40.4350891, -108.1055374, 108.1714859
20: -68.5305328, 53.0863914, -68.6430969, 53.1378250, -121.6683578, 121.7294922
21: -85.0669785, 51.0934486, -85.2700043, 51.2065277, -136.2734985, 136.3634491
22: -86.5905533, 46.4214401, -86.7447205, 46.5170670, -133.1076202, 133.1661682
23: -70.0036850, 54.0150833, -70.1219635, 54.0882034, -124.0918884, 124.1370392
24: -90.3324509, 54.4058647, -90.4485931, 54.5060501, -144.8384857, 144.8544617
25: -76.1137390, 55.4565430, -76.2168427, 55.5520287, -131.6657715, 131.6733856
26: -100.8341446, 81.9402466, -101.0356750, 82.1310043, -182.9651489, 182.9759216
27: -87.7478027, 49.5305710, -87.9514084, 49.6506691, -137.3984680, 137.4819794
28: -68.4231949, 54.3958473, -68.5552368, 54.5105438, -122.9337387, 122.9510803
29: -89.1385880, 41.6981583, -89.2813797, 41.7922363, -130.9308167, 130.9795227
30: -88.3236847, 63.6664124, -88.4336777, 63.7552414, -152.0789185, 152.1000977
31: -91.4931259, 55.9940186, -91.6429138, 56.0709648, -147.5640869, 147.6369324
32: -89.9597626, 57.4589539, -90.1098709, 57.5209427, -147.4806824, 147.5688171
33: -126.7408295, 77.8821869, -126.9691162, 78.1736755, -204.9145050, 204.8512878
34: -106.2507324, 48.5847855, -106.4813385, 48.8192596, -155.0699768, 155.0661316
35: -99.0961761, 58.6760941, -99.3042145, 58.9540329, -158.0501862, 157.9803009
36: -92.4875031, 57.1874619, -92.7118454, 57.3694992, -149.8569946, 149.8992920
37: -145.5373383, 62.7311172, -145.7367249, 62.9035454, -208.4408417, 208.4678345
38: -112.2578125, 71.1898727, -112.5299759, 71.4515305, -183.7093506, 183.7198486
39: -133.2735596, 76.5181274, -133.4711456, 76.7085342, -209.9820862, 209.9892731
40: -111.0020142, 56.7992134, -111.2038116, 56.9573898, -167.9593964, 168.0030212
41: -95.8308868, 65.8236084, -96.0164108, 65.9422684, -161.7731323, 161.8400269
42: -70.2700653, 56.5845528, -70.4143982, 56.6778221, -126.9478683, 126.9989395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1669
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
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 878
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
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1528
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.2483103, upper bound: 93.5711279
time: 89.29 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -93.4921194, upper bound: 93.5711278
time: 242.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 333.99 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.3455116, upper bound: 93.4749919
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.3455115, upper bound: 93.4749919
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.3455104, upper bound: 93.5904666
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.3455104, upper bound: 93.5904666
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.3455116, upper bound: 93.4122011
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.3455116, upper bound: 93.4122011
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.3455104, upper bound: 93.4916390
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.3455104, upper bound: 93.4916390
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.2472191, upper bound: 93.5711276
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.2472191, upper bound: 93.5711276
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.2472191, upper bound: 93.7189853
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.2472191, upper bound: 93.7189853
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.2483103, upper bound: 93.5711279
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 333.99
Output dim: 9, lower bound: -93.4921194, upper bound: 93.5711278
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 333.99
Output dim: 9, lower bound: -93.4998451, upper bound: 93.7271037
Binary search (step 0): status=Status.UNKNOWN, k_low=8, k_high=12, k_mid=10, eps_mid=0.0390625, abs_max=137.98736572265625
rel_dist={9: [-93.75246575465076, 93.75246575465076]}

## Binary search (step 1) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1685

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.5645249, upper bound: 91.3827895
time: 528.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.5645249, upper bound: 91.5645247
time: 114.21 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 642.92 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 642.92
Output dim: 9, lower bound: -91.5645249, upper bound: 91.3827895
IS_A2, status: Status.UNKNOWN, split count: 1, time: 642.92
Output dim: 9, lower bound: -91.5645249, upper bound: 91.5645247

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -108.6825104, 78.8020706, -108.8903046, 78.9298248, -187.6123199, 187.6923676
1: -57.0678215, 59.0297890, -57.1893463, 59.1159859, -116.1837921, 116.2191315
2: -49.3603210, 60.4166794, -49.4568939, 60.5051880, -109.8655090, 109.8735733
3: -62.0270157, 73.3680878, -62.1418495, 73.5388489, -135.5658569, 135.5099335
4: -64.6978760, 70.6075897, -64.8346481, 70.7556763, -135.4535522, 135.4422302
5: -59.6082916, 73.0030823, -59.7510185, 73.1751251, -132.7834167, 132.7540894
6: -94.2190552, 62.8569450, -94.4204636, 62.9671135, -157.1861572, 157.2773895
7: -66.5231934, 69.4289551, -66.6955414, 69.5656281, -136.0888214, 136.1244965
8: -81.3496399, 83.7510376, -81.4543152, 83.8826828, -165.2323303, 165.2053528
9: -60.6766510, 76.6171570, -60.9909592, 76.9062500, -137.5829010, 137.6081238
10: -88.3650818, 90.9195251, -88.8892517, 91.3758240, -179.7409058, 179.8087769
11: -85.7005463, 58.0707436, -86.0588074, 58.2340431, -143.9345856, 144.1295471
12: -97.5419464, 77.0192108, -97.7228088, 77.1426468, -174.6845856, 174.7420197
13: -84.9839172, 98.4149780, -85.1263733, 98.5697784, -183.5536804, 183.5413513
14: -144.1477051, 82.0576172, -144.5269470, 82.3632812, -226.5109863, 226.5845642
15: -78.6271057, 64.2450562, -78.7351456, 64.3925858, -143.0196838, 142.9801941
16: -90.9301682, 72.2346725, -91.3546295, 72.5360031, -163.4661713, 163.5892944
17: -133.2652130, 71.6172714, -133.6279602, 71.8105240, -205.0757446, 205.2452087
18: -93.2614899, 69.7886658, -93.4541702, 69.9037018, -163.1651764, 163.2428284
19: -67.6487885, 40.3830147, -67.8096390, 40.4372025, -108.0859833, 108.1926575
20: -68.4995728, 53.0521393, -68.6391144, 53.1329041, -121.6324768, 121.6912537
21: -85.0168381, 51.1076126, -85.2615509, 51.2081184, -136.2249451, 136.3691711
22: -86.4652023, 46.3651047, -86.7186661, 46.5150757, -132.9802856, 133.0837708
23: -69.9554596, 54.0113220, -70.1161423, 54.0873528, -124.0428162, 124.1274643
24: -90.2603989, 54.3981781, -90.4477158, 54.5069199, -144.7673187, 144.8458862
25: -76.0829163, 55.4306641, -76.2096100, 55.5497398, -131.6326599, 131.6402588
26: -100.8495712, 82.0273743, -101.0468292, 82.1404190, -182.9899902, 183.0742035
27: -87.7463531, 49.5539932, -87.9585648, 49.6513443, -137.3977051, 137.5125580
28: -68.4534607, 54.4348679, -68.5588531, 54.5090561, -122.9625092, 122.9937134
29: -89.0571289, 41.7203522, -89.2702484, 41.7954407, -130.8525543, 130.9906006
30: -88.2577209, 63.5894547, -88.4288788, 63.7407379, -151.9984589, 152.0183411
31: -91.4716110, 56.0199738, -91.6449585, 56.0744438, -147.5460510, 147.6649323
32: -89.9175873, 57.4322014, -90.1004181, 57.5195389, -147.4371185, 147.5326233
33: -126.6107407, 77.8046570, -126.9308777, 78.1734772, -204.7842102, 204.7355347
34: -106.3053589, 48.5926857, -106.4829178, 48.8165588, -155.1219177, 155.0755920
35: -99.0683594, 58.6328049, -99.2897720, 58.9523544, -158.0207062, 157.9225616
36: -92.4493256, 57.0919380, -92.6918945, 57.3671494, -149.8164673, 149.7838287
37: -145.2531738, 62.5798264, -145.6751556, 62.9031029, -208.1562500, 208.2549744
38: -112.1943970, 71.0787964, -112.5048370, 71.4489594, -183.6433563, 183.5836182
39: -133.0400085, 76.3493347, -133.4194031, 76.7088776, -209.7488403, 209.7687378
40: -110.8761749, 56.7141113, -111.1722946, 56.9559212, -167.8320923, 167.8864136
41: -95.7493744, 65.7828827, -95.9920349, 65.9430237, -161.6923828, 161.7749176
42: -70.2251740, 56.6016235, -70.4017029, 56.6971359, -126.9223099, 127.0033264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=522, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1685
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
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 656
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
type: B, layer: 1, pos: 934
type: B, layer: 1, pos: 847
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
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 789
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
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1528
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
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1036

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1685

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.3827895, upper bound: 91.3827895
time: 109.25 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.3827895, upper bound: 91.3827895
time: 159.05 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -108.9021606, 78.9381256, -108.9130936, 78.9541168, -187.8562775, 187.8512268
1: -57.1944237, 59.1250954, -57.1987114, 59.1296959, -116.3241196, 116.3238068
2: -49.4602089, 60.5124817, -49.4704895, 60.5170021, -109.9772110, 109.9829712
3: -62.1488037, 73.5440979, -62.1570511, 73.5699692, -135.7187805, 135.7011414
4: -64.8313599, 70.7698593, -64.8567657, 70.7765961, -135.6079559, 135.6266174
5: -59.7581177, 73.1889801, -59.7647324, 73.2092133, -132.9673309, 132.9537048
6: -94.4425659, 62.9719315, -94.4595261, 62.9757347, -157.4183044, 157.4314423
7: -66.7029877, 69.5845032, -66.7078934, 69.5958099, -136.2987976, 136.2923889
8: -81.4612885, 83.8939972, -81.4663239, 83.9004364, -165.3616943, 165.3603210
9: -60.9977417, 76.9633636, -61.0038757, 76.9798584, -137.9775848, 137.9672394
10: -88.8991547, 91.4643250, -88.9056702, 91.4937897, -180.3929443, 180.3699951
11: -86.0711594, 58.2591438, -86.0769348, 58.2721977, -144.3433533, 144.3360748
12: -97.7410355, 77.1502075, -97.7495575, 77.1568451, -174.8978882, 174.8997650
13: -85.1333160, 98.5814056, -85.1442490, 98.5884399, -183.7217560, 183.7256470
14: -144.5381165, 82.4331665, -144.5480652, 82.4486389, -226.9867554, 226.9812317
15: -78.7409515, 64.4085541, -78.7479401, 64.4156799, -143.1566162, 143.1564941
16: -91.3679581, 72.5947571, -91.3767548, 72.6137695, -163.9817200, 163.9715118
17: -133.6418152, 71.8285294, -133.6496582, 71.8364410, -205.4782562, 205.4781799
18: -93.4431305, 69.9135056, -93.4766006, 69.9229889, -163.3661194, 163.3901062
19: -67.8218231, 40.4408417, -67.8300629, 40.4460602, -108.2678833, 108.2709045
20: -68.6486969, 53.1403923, -68.6545105, 53.1470604, -121.7957611, 121.7949066
21: -85.2771988, 51.2163391, -85.2848511, 51.2267227, -136.5039215, 136.5011902
22: -86.7549820, 46.5233116, -86.7681122, 46.5285606, -133.2835236, 133.2914124
23: -70.1270905, 54.0933571, -70.1336670, 54.1013260, -124.2284088, 124.2270126
24: -90.4580688, 54.5093765, -90.4881744, 54.5165977, -144.9746399, 144.9975433
25: -76.2221527, 55.5569839, -76.2281952, 55.5629463, -131.7850952, 131.7851868
26: -101.0425262, 82.1481018, -101.0725098, 82.1579895, -183.2005005, 183.2206116
27: -87.9699097, 49.6536369, -88.0012512, 49.6608658, -137.6307678, 137.6548920
28: -68.5663452, 54.5140762, -68.5751038, 54.5187378, -123.0850830, 123.0891800
29: -89.2900162, 41.8007965, -89.3016205, 41.8059006, -131.0959015, 131.1024170
30: -88.4397125, 63.7614975, -88.4464111, 63.7714462, -152.2111511, 152.2079163
31: -91.6505814, 56.0764847, -91.6687698, 56.0841446, -147.7347260, 147.7452545
32: -90.1213760, 57.5232277, -90.1371689, 57.5264473, -147.6478271, 147.6603851
33: -126.9933014, 78.1782227, -127.0152359, 78.1824799, -205.1757507, 205.1934509
34: -106.5051193, 48.8233833, -106.5246964, 48.8269157, -155.3320312, 155.3480682
35: -99.3263855, 58.9572334, -99.3443909, 58.9599533, -158.2863464, 158.3016205
36: -92.7336502, 57.3715973, -92.7524109, 57.3734207, -150.1070709, 150.1240082
37: -145.7526855, 62.9080963, -145.7778931, 62.9123268, -208.6650085, 208.6859894
38: -112.5565109, 71.4561234, -112.5777283, 71.4611359, -184.0176392, 184.0338440
39: -133.4869385, 76.7122040, -133.5100861, 76.7166519, -210.2035828, 210.2222900
40: -111.2218399, 56.9596291, -111.2395325, 56.9624481, -168.1842804, 168.1991577
41: -96.0328751, 65.9454193, -96.0490265, 65.9491806, -161.9820557, 161.9944458
42: -70.4235611, 56.6825485, -70.4332886, 56.7060394, -127.1296005, 127.1158371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=522, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1685
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
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 857
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
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1626
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
type: B, layer: 1, pos: 1233
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1024
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1036

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1685

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.3827895, upper bound: 91.5645249
time: 181.70 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.3827895, upper bound: 91.5645249
time: 144.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 328.95 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 328.95
Output dim: 9, lower bound: -91.3827895, upper bound: 91.3827895
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 328.95
Output dim: 9, lower bound: -91.3827895, upper bound: 91.3827895
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 328.95
Output dim: 9, lower bound: -91.3827895, upper bound: 91.5645249
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 328.95
Output dim: 9, lower bound: -91.3827895, upper bound: 91.5645249

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -108.6825104, 78.8020706, -108.6825104, 78.8020706, -187.4845734, 187.4845734
1: -57.0678215, 59.0297890, -57.0678215, 59.0297890, -116.0976105, 116.0976105
2: -49.3603210, 60.4166794, -49.3603210, 60.4166794, -109.7769928, 109.7770004
3: -62.0270157, 73.3680878, -62.0270157, 73.3680878, -135.3951111, 135.3950958
4: -64.6978760, 70.6075897, -64.6978760, 70.6075897, -135.3054504, 135.3054504
5: -59.6082916, 73.0030823, -59.6082916, 73.0030823, -132.6113739, 132.6113739
6: -94.2190552, 62.8569450, -94.2190552, 62.8569450, -157.0759888, 157.0759888
7: -66.5231934, 69.4289551, -66.5231934, 69.4289551, -135.9521484, 135.9521484
8: -81.3496399, 83.7510376, -81.3496399, 83.7510376, -165.1006775, 165.1006775
9: -60.6766510, 76.6171570, -60.6766510, 76.6171570, -137.2938080, 137.2938080
10: -88.3650818, 90.9195251, -88.3650818, 90.9195251, -179.2845917, 179.2845917
11: -85.7005463, 58.0707436, -85.7005463, 58.0707436, -143.7712860, 143.7712860
12: -97.5419464, 77.0192108, -97.5419464, 77.0192108, -174.5611420, 174.5611420
13: -84.9839172, 98.4149780, -84.9839172, 98.4149780, -183.3988953, 183.3988953
14: -144.1477051, 82.0576172, -144.1477051, 82.0576172, -226.2052917, 226.2053223
15: -78.6271057, 64.2450562, -78.6271057, 64.2450562, -142.8721619, 142.8721619
16: -90.9301682, 72.2346725, -90.9301682, 72.2346725, -163.1648407, 163.1648407
17: -133.2652130, 71.6172714, -133.2652130, 71.6172714, -204.8824768, 204.8824768
18: -93.2614899, 69.7886658, -93.2614899, 69.7886658, -163.0501404, 163.0501556
19: -67.6487885, 40.3830147, -67.6487885, 40.3830147, -108.0317993, 108.0317993
20: -68.4995728, 53.0521393, -68.4995728, 53.0521393, -121.5517120, 121.5517120
21: -85.0168381, 51.1076126, -85.0168381, 51.1076126, -136.1244507, 136.1244507
22: -86.4652023, 46.3651047, -86.4652023, 46.3651047, -132.8303070, 132.8303070
23: -69.9554596, 54.0113220, -69.9554596, 54.0113220, -123.9667816, 123.9667816
24: -90.2603989, 54.3981781, -90.2603989, 54.3981781, -144.6585693, 144.6585693
25: -76.0829163, 55.4306641, -76.0829163, 55.4306641, -131.5135803, 131.5135803
26: -100.8495712, 82.0273743, -100.8495712, 82.0273743, -182.8769379, 182.8769531
27: -87.7463531, 49.5539932, -87.7463531, 49.5539932, -137.3003540, 137.3003540
28: -68.4534607, 54.4348679, -68.4534607, 54.4348679, -122.8883133, 122.8883133
29: -89.0571289, 41.7203522, -89.0571289, 41.7203522, -130.7774811, 130.7774811
30: -88.2577209, 63.5894547, -88.2577209, 63.5894547, -151.8471680, 151.8471680
31: -91.4716110, 56.0199738, -91.4716110, 56.0199738, -147.4915771, 147.4915771
32: -89.9175873, 57.4322014, -89.9175873, 57.4322014, -147.3497772, 147.3497772
33: -126.6107407, 77.8046570, -126.6107407, 77.8046570, -204.4154053, 204.4153900
34: -106.3053589, 48.5926857, -106.3053589, 48.5926857, -154.8980255, 154.8980408
35: -99.0683594, 58.6328049, -99.0683594, 58.6328049, -157.7011566, 157.7011719
36: -92.4493256, 57.0919380, -92.4493256, 57.0919380, -149.5412598, 149.5412598
37: -145.2531738, 62.5798264, -145.2531738, 62.5798264, -207.8329773, 207.8330078
38: -112.1943970, 71.0787964, -112.1943970, 71.0787964, -183.2731934, 183.2731628
39: -133.0400085, 76.3493347, -133.0400085, 76.3493347, -209.3893280, 209.3893433
40: -110.8761749, 56.7141113, -110.8761749, 56.7141113, -167.5902863, 167.5902863
41: -95.7493744, 65.7828827, -95.7493744, 65.7828827, -161.5322571, 161.5322266
42: -70.2251740, 56.6016235, -70.2251740, 56.6016235, -126.8267975, 126.8267975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 856
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
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1464
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
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1738
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
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1669

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.4638692, upper bound: 91.2989359
time: 100.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.4638692, upper bound: 91.3680005
time: 108.15 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -108.6825104, 78.8020706, -108.9021606, 78.9381256, -187.6206360, 187.7042236
1: -57.0678215, 59.0297890, -57.1944237, 59.1250954, -116.1929169, 116.2242126
2: -49.3603210, 60.4166794, -49.4602089, 60.5124817, -109.8728027, 109.8768921
3: -62.0270157, 73.3680878, -62.1488037, 73.5440979, -135.5711060, 135.5168915
4: -64.6978760, 70.6075897, -64.8313599, 70.7698593, -135.4677124, 135.4389496
5: -59.6082916, 73.0030823, -59.7581177, 73.1889801, -132.7972717, 132.7612000
6: -94.2190552, 62.8569450, -94.4425659, 62.9719315, -157.1909790, 157.2994995
7: -66.5231934, 69.4289551, -66.7029877, 69.5845032, -136.1076965, 136.1319427
8: -81.3496399, 83.7510376, -81.4612885, 83.8939972, -165.2436218, 165.2123108
9: -60.6766510, 76.6171570, -60.9977417, 76.9633636, -137.6400146, 137.6148987
10: -88.3650818, 90.9195251, -88.8991547, 91.4643250, -179.8294067, 179.8186646
11: -85.7005463, 58.0707436, -86.0711594, 58.2591438, -143.9596863, 144.1419067
12: -97.5419464, 77.0192108, -97.7410355, 77.1502075, -174.6921539, 174.7602386
13: -84.9839172, 98.4149780, -85.1333160, 98.5814056, -183.5653229, 183.5482941
14: -144.1477051, 82.0576172, -144.5381165, 82.4331665, -226.5808716, 226.5957336
15: -78.6271057, 64.2450562, -78.7409515, 64.4085541, -143.0356598, 142.9860077
16: -90.9301682, 72.2346725, -91.3679581, 72.5947571, -163.5249176, 163.6026306
17: -133.2652130, 71.6172714, -133.6418152, 71.8285294, -205.0937500, 205.2590790
18: -93.2614899, 69.7886658, -93.4431305, 69.9135056, -163.1749725, 163.2317810
19: -67.6487885, 40.3830147, -67.8218231, 40.4408417, -108.0896301, 108.2048340
20: -68.4995728, 53.0521393, -68.6486969, 53.1403923, -121.6399536, 121.7008362
21: -85.0168381, 51.1076126, -85.2771988, 51.2163391, -136.2331696, 136.3848114
22: -86.4652023, 46.3651047, -86.7549820, 46.5233116, -132.9885101, 133.1200867
23: -69.9554596, 54.0113220, -70.1270905, 54.0933571, -124.0488129, 124.1384125
24: -90.2603989, 54.3981781, -90.4580688, 54.5093765, -144.7697754, 144.8562317
25: -76.0829163, 55.4306641, -76.2221527, 55.5569839, -131.6398926, 131.6528015
26: -100.8495712, 82.0273743, -101.0425262, 82.1481018, -182.9976807, 183.0699005
27: -87.7463531, 49.5539932, -87.9699097, 49.6536369, -137.3999939, 137.5238953
28: -68.4534607, 54.4348679, -68.5663452, 54.5140762, -122.9675293, 123.0012131
29: -89.0571289, 41.7203522, -89.2900162, 41.8007965, -130.8579254, 131.0103760
30: -88.2577209, 63.5894547, -88.4397125, 63.7614975, -152.0192108, 152.0291748
31: -91.4716110, 56.0199738, -91.6505814, 56.0764847, -147.5480957, 147.6705627
32: -89.9175873, 57.4322014, -90.1213760, 57.5232277, -147.4407959, 147.5535736
33: -126.6107407, 77.8046570, -126.9933014, 78.1782227, -204.7889404, 204.7979431
34: -106.3053589, 48.5926857, -106.5051193, 48.8233833, -155.1287231, 155.0977936
35: -99.0683594, 58.6328049, -99.3263855, 58.9572334, -158.0255890, 157.9591980
36: -92.4493256, 57.0919380, -92.7336502, 57.3715973, -149.8209229, 149.8255920
37: -145.2531738, 62.5798264, -145.7526855, 62.9080963, -208.1612549, 208.3325195
38: -112.1943970, 71.0787964, -112.5565109, 71.4561234, -183.6505127, 183.6352844
39: -133.0400085, 76.3493347, -133.4869385, 76.7122040, -209.7522125, 209.8362732
40: -110.8761749, 56.7141113, -111.2218399, 56.9596291, -167.8358002, 167.9359436
41: -95.7493744, 65.7828827, -96.0328751, 65.9454193, -161.6947937, 161.8157349
42: -70.2251740, 56.6016235, -70.4235611, 56.6825485, -126.9077225, 127.0251846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=521, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 856
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
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1464
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
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1738
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
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1233
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1024
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1036

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1669

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.4638692, upper bound: 91.2989359
time: 212.16 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.4638692, upper bound: 91.3680005
time: 219.13 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -108.9021606, 78.9381256, -108.6825104, 78.8020706, -187.7042084, 187.6206360
1: -57.1944237, 59.1250954, -57.0678215, 59.0297890, -116.2242126, 116.1929092
2: -49.4602089, 60.5124817, -49.3603210, 60.4166794, -109.8768921, 109.8728027
3: -62.1488037, 73.5440979, -62.0270157, 73.3680878, -135.5168915, 135.5711060
4: -64.8313599, 70.7698593, -64.6978760, 70.6075897, -135.4389496, 135.4677124
5: -59.7581177, 73.1889801, -59.6082916, 73.0030823, -132.7612000, 132.7972717
6: -94.4425659, 62.9719315, -94.2190552, 62.8569450, -157.2994995, 157.1909790
7: -66.7029877, 69.5845032, -66.5231934, 69.4289551, -136.1319427, 136.1076965
8: -81.4612885, 83.8939972, -81.3496399, 83.7510376, -165.2123260, 165.2436371
9: -60.9977417, 76.9633636, -60.6766510, 76.6171570, -137.6148987, 137.6400146
10: -88.8991547, 91.4643250, -88.3650818, 90.9195251, -179.8186798, 179.8294067
11: -86.0711594, 58.2591438, -85.7005463, 58.0707436, -144.1419067, 143.9596863
12: -97.7410355, 77.1502075, -97.5419464, 77.0192108, -174.7602386, 174.6921539
13: -85.1333160, 98.5814056, -84.9839172, 98.4149780, -183.5482788, 183.5653076
14: -144.5381165, 82.4331665, -144.1477051, 82.0576172, -226.5957336, 226.5808716
15: -78.7409515, 64.4085541, -78.6271057, 64.2450562, -142.9860077, 143.0356598
16: -91.3679581, 72.5947571, -90.9301682, 72.2346725, -163.6026306, 163.5249023
17: -133.6418152, 71.8285294, -133.2652130, 71.6172714, -205.2590942, 205.0937500
18: -93.4431305, 69.9135056, -93.2614899, 69.7886658, -163.2317810, 163.1749878
19: -67.8218231, 40.4408417, -67.6487885, 40.3830147, -108.2048340, 108.0896301
20: -68.6486969, 53.1403923, -68.4995728, 53.0521393, -121.7008209, 121.6399536
21: -85.2771988, 51.2163391, -85.0168381, 51.1076126, -136.3847961, 136.2331848
22: -86.7549820, 46.5233116, -86.4652023, 46.3651047, -133.1200867, 132.9885101
23: -70.1270905, 54.0933571, -69.9554596, 54.0113220, -124.1384125, 124.0488129
24: -90.4580688, 54.5093765, -90.2603989, 54.3981781, -144.8562469, 144.7697754
25: -76.2221527, 55.5569839, -76.0829163, 55.4306641, -131.6528015, 131.6398926
26: -101.0425262, 82.1481018, -100.8495712, 82.0273743, -183.0699005, 182.9976807
27: -87.9699097, 49.6536369, -87.7463531, 49.5539932, -137.5238953, 137.3999939
28: -68.5663452, 54.5140762, -68.4534607, 54.4348679, -123.0012131, 122.9675217
29: -89.2900162, 41.8007965, -89.0571289, 41.7203522, -131.0103760, 130.8579254
30: -88.4397125, 63.7614975, -88.2577209, 63.5894547, -152.0291748, 152.0192261
31: -91.6505814, 56.0764847, -91.4716110, 56.0199738, -147.6705475, 147.5480957
32: -90.1213760, 57.5232277, -89.9175873, 57.4322014, -147.5535736, 147.4407959
33: -126.9933014, 78.1782227, -126.6107407, 77.8046570, -204.7979431, 204.7889557
34: -106.5051193, 48.8233833, -106.3053589, 48.5926857, -155.0977936, 155.1287384
35: -99.3263855, 58.9572334, -99.0683594, 58.6328049, -157.9591980, 158.0255890
36: -92.7336502, 57.3715973, -92.4493256, 57.0919380, -149.8255920, 149.8209229
37: -145.7526855, 62.9080963, -145.2531738, 62.5798264, -208.3324890, 208.1612701
38: -112.5565109, 71.4561234, -112.1943970, 71.0787964, -183.6352692, 183.6505127
39: -133.4869385, 76.7122040, -133.0400085, 76.3493347, -209.8362732, 209.7522125
40: -111.2218399, 56.9596291, -110.8761749, 56.7141113, -167.9359436, 167.8358002
41: -96.0328751, 65.9454193, -95.7493744, 65.7828827, -161.8157349, 161.6947937
42: -70.4235611, 56.6825485, -70.2251740, 56.6016235, -127.0251846, 126.9077225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=521, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1669

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.3680006, upper bound: 91.4294947
time: 112.51 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.3680006, upper bound: 91.5558787
time: 109.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -108.9021606, 78.9381256, -108.9021606, 78.9381256, -187.8402710, 187.8402863
1: -57.1944237, 59.1250954, -57.1944237, 59.1250954, -116.3195190, 116.3195190
2: -49.4602089, 60.5124817, -49.4602089, 60.5124817, -109.9726868, 109.9726868
3: -62.1488037, 73.5440979, -62.1488037, 73.5440979, -135.6929016, 135.6929016
4: -64.8313599, 70.7698593, -64.8313599, 70.7698593, -135.6012268, 135.6012268
5: -59.7581177, 73.1889801, -59.7581177, 73.1889801, -132.9470978, 132.9470978
6: -94.4425659, 62.9719315, -94.4425659, 62.9719315, -157.4144897, 157.4144897
7: -66.7029877, 69.5845032, -66.7029877, 69.5845032, -136.2874908, 136.2874908
8: -81.4612885, 83.8939972, -81.4612885, 83.8939972, -165.3552551, 165.3552856
9: -60.9977417, 76.9633636, -60.9977417, 76.9633636, -137.9611053, 137.9611053
10: -88.8991547, 91.4643250, -88.8991547, 91.4643250, -180.3634796, 180.3634644
11: -86.0711594, 58.2591438, -86.0711594, 58.2591438, -144.3302917, 144.3303070
12: -97.7410355, 77.1502075, -97.7410355, 77.1502075, -174.8912354, 174.8912354
13: -85.1333160, 98.5814056, -85.1333160, 98.5814056, -183.7147064, 183.7147064
14: -144.5381165, 82.4331665, -144.5381165, 82.4331665, -226.9712830, 226.9712830
15: -78.7409515, 64.4085541, -78.7409515, 64.4085541, -143.1495056, 143.1495056
16: -91.3679581, 72.5947571, -91.3679581, 72.5947571, -163.9627075, 163.9627075
17: -133.6418152, 71.8285294, -133.6418152, 71.8285294, -205.4703369, 205.4703369
18: -93.4431305, 69.9135056, -93.4431305, 69.9135056, -163.3566284, 163.3566284
19: -67.8218231, 40.4408417, -67.8218231, 40.4408417, -108.2626648, 108.2626648
20: -68.6486969, 53.1403923, -68.6486969, 53.1403923, -121.7890930, 121.7890854
21: -85.2771988, 51.2163391, -85.2771988, 51.2163391, -136.4935303, 136.4935303
22: -86.7549820, 46.5233116, -86.7549820, 46.5233116, -133.2782898, 133.2782898
23: -70.1270905, 54.0933571, -70.1270905, 54.0933571, -124.2204361, 124.2204361
24: -90.4580688, 54.5093765, -90.4580688, 54.5093765, -144.9674377, 144.9674225
25: -76.2221527, 55.5569839, -76.2221527, 55.5569839, -131.7791443, 131.7791290
26: -101.0425262, 82.1481018, -101.0425262, 82.1481018, -183.1906128, 183.1906281
27: -87.9699097, 49.6536369, -87.9699097, 49.6536369, -137.6235504, 137.6235504
28: -68.5663452, 54.5140762, -68.5663452, 54.5140762, -123.0804214, 123.0804214
29: -89.2900162, 41.8007965, -89.2900162, 41.8007965, -131.0908203, 131.0908203
30: -88.4397125, 63.7614975, -88.4397125, 63.7614975, -152.2012024, 152.2012024
31: -91.6505814, 56.0764847, -91.6505814, 56.0764847, -147.7270660, 147.7270508
32: -90.1213760, 57.5232277, -90.1213760, 57.5232277, -147.6446075, 147.6446075
33: -126.9933014, 78.1782227, -126.9933014, 78.1782227, -205.1715088, 205.1715088
34: -106.5051193, 48.8233833, -106.5051193, 48.8233833, -155.3284912, 155.3285065
35: -99.3263855, 58.9572334, -99.3263855, 58.9572334, -158.2836151, 158.2836151
36: -92.7336502, 57.3715973, -92.7336502, 57.3715973, -150.1052399, 150.1052551
37: -145.7526855, 62.9080963, -145.7526855, 62.9080963, -208.6607666, 208.6607819
38: -112.5565109, 71.4561234, -112.5565109, 71.4561234, -184.0126038, 184.0126343
39: -133.4869385, 76.7122040, -133.4869385, 76.7122040, -210.1991272, 210.1991425
40: -111.2218399, 56.9596291, -111.2218399, 56.9596291, -168.1814728, 168.1814728
41: -96.0328751, 65.9454193, -96.0328751, 65.9454193, -161.9782715, 161.9782715
42: -70.4235611, 56.6825485, -70.4235611, 56.6825485, -127.1061096, 127.1061096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=521, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1669

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.3680006, upper bound: 91.4294950
time: 95.83 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.3680006, upper bound: 91.5558791
time: 95.50 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 193.70 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 193.70
Output dim: 9, lower bound: -91.4638692, upper bound: 91.2989359
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 193.70
Output dim: 9, lower bound: -91.4638692, upper bound: 91.3680005
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 193.70
Output dim: 9, lower bound: -91.4638692, upper bound: 91.2989359
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 193.70
Output dim: 9, lower bound: -91.4638692, upper bound: 91.3680005
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 193.70
Output dim: 9, lower bound: -91.3680006, upper bound: 91.4294947
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 193.70
Output dim: 9, lower bound: -91.3680006, upper bound: 91.5558787
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 193.70
Output dim: 9, lower bound: -91.3680006, upper bound: 91.4294950
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 193.70
Output dim: 9, lower bound: -91.3680006, upper bound: 91.5558791

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -108.5506516, 78.7246170, -108.6690521, 78.7942505, -187.3448792, 187.3936768
1: -56.9843712, 58.9458046, -57.0592194, 59.0224609, -116.0068359, 116.0050201
2: -49.2399521, 60.3107491, -49.3412247, 60.4088745, -109.6488037, 109.6519623
3: -61.8565826, 73.2091599, -61.9987259, 73.3589706, -135.2155457, 135.2078857
4: -64.5778809, 70.4876480, -64.6818085, 70.5982056, -135.1760864, 135.1694641
5: -59.4774971, 72.8726349, -59.5894737, 72.9942017, -132.4716949, 132.4620972
6: -93.9823608, 62.7458153, -94.1814423, 62.8509026, -156.8332672, 156.9272461
7: -66.3727417, 69.3187027, -66.5033722, 69.4212036, -135.7939453, 135.8220825
8: -81.2441406, 83.6334381, -81.3367004, 83.7418213, -164.9859619, 164.9701385
9: -60.4303360, 76.3844223, -60.6669464, 76.5730896, -137.0034180, 137.0513611
10: -87.8765335, 90.4492493, -88.3526382, 90.8291626, -178.7056885, 178.8018799
11: -85.4578476, 57.9242706, -85.6899490, 58.0452118, -143.5030518, 143.6142273
12: -97.2437210, 76.7960815, -97.5293655, 76.9793625, -174.2230682, 174.3254395
13: -84.8233566, 98.2853851, -84.9687271, 98.4008026, -183.2241516, 183.2541046
14: -143.7672119, 81.7253189, -144.1305847, 81.9932709, -225.7604828, 225.8558960
15: -78.4989471, 64.0957718, -78.6142654, 64.2259293, -142.7248688, 142.7100372
16: -90.6548080, 72.0003891, -90.9149094, 72.1932297, -162.8480225, 162.9152832
17: -133.0215454, 71.4710007, -133.2538452, 71.5968552, -204.6184082, 204.7248230
18: -93.1057663, 69.6566620, -93.2477875, 69.7731476, -162.8789062, 162.9044495
19: -67.4990768, 40.2989426, -67.6362991, 40.3718071, -107.8708725, 107.9352417
20: -68.3830109, 52.9986687, -68.4886780, 53.0471573, -121.4301682, 121.4873352
21: -84.8121872, 50.9863510, -85.0028687, 51.0886803, -135.9008636, 135.9892273
22: -86.3031616, 46.2641792, -86.4456177, 46.3527145, -132.6558685, 132.7097931
23: -69.8331451, 53.9347229, -69.9455414, 54.0015793, -123.8347015, 123.8802490
24: -90.1385345, 54.3005524, -90.2428284, 54.3917007, -144.5302429, 144.5433655
25: -75.9751282, 55.3318787, -76.0725098, 55.4209442, -131.3960724, 131.4043884
26: -100.6410675, 81.8202820, -100.8362961, 81.9938049, -182.6348724, 182.6565857
27: -87.5329666, 49.4462128, -87.7120514, 49.5481529, -137.0811157, 137.1582642
28: -68.3119354, 54.3198624, -68.4321365, 54.4279938, -122.7399139, 122.7519913
29: -88.9080734, 41.6175652, -89.0405884, 41.7033157, -130.6113892, 130.6581573
30: -88.1417923, 63.4941139, -88.2457886, 63.5774307, -151.7192078, 151.7398987
31: -91.3166199, 55.9384460, -91.4567490, 56.0093651, -147.3259735, 147.3952026
32: -89.7597885, 57.3673592, -89.8956451, 57.4276733, -147.1874695, 147.2630005
33: -126.3591843, 77.5131989, -126.5638504, 77.7959442, -204.1551208, 204.0770569
34: -106.0513229, 48.3582001, -106.2592392, 48.5845490, -154.6358643, 154.6174316
35: -98.8432693, 58.3575058, -99.0262833, 58.6264839, -157.4697266, 157.3837891
36: -92.2039261, 56.9101715, -92.4068985, 57.0878448, -149.2917786, 149.3170624
37: -145.0395966, 62.4041977, -145.2222900, 62.5707817, -207.6103821, 207.6264801
38: -111.8955765, 70.8134384, -112.1424713, 71.0696945, -182.9652710, 182.9559021
39: -132.8282166, 76.1567230, -133.0094910, 76.3422699, -209.1704865, 209.1662140
40: -110.6603088, 56.5544662, -110.8414917, 56.7097282, -167.3700256, 167.3959656
41: -95.5523834, 65.6622238, -95.7178802, 65.7766571, -161.3290405, 161.3800964
42: -70.0766296, 56.5032234, -70.2072906, 56.5923538, -126.6689835, 126.7105103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1686
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
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1699
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
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1689
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
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 956
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
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 934
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
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 703
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
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 537
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
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1031
type: B, layer: 1, pos: 1760
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
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1649
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.2520455, upper bound: 91.3467921
time: 101.44 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.2520455, upper bound: 91.3467921
time: 158.00 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -108.6696320, 78.7972336, -108.6800613, 78.8011169, -187.4707489, 187.4772949
1: -57.0531540, 59.0252686, -57.0650826, 59.0288887, -116.0820389, 116.0903473
2: -49.3495026, 60.4134102, -49.3582993, 60.4160156, -109.7655106, 109.7717056
3: -62.0051155, 73.3640747, -62.0224915, 73.3673172, -135.3724365, 135.3865662
4: -64.6748352, 70.6014099, -64.6936493, 70.6063385, -135.2811737, 135.2950592
5: -59.5821762, 72.9992371, -59.6030769, 73.0023651, -132.5845337, 132.6023102
6: -94.2035522, 62.8539047, -94.2161102, 62.8563309, -157.0598755, 157.0700073
7: -66.5003357, 69.4254608, -66.5189590, 69.4282684, -135.9285889, 135.9444122
8: -81.3386002, 83.7470093, -81.3472900, 83.7502441, -165.0888062, 165.0942993
9: -60.6712837, 76.6020432, -60.6756134, 76.6142960, -137.2855835, 137.2776489
10: -88.3595886, 90.8887634, -88.3640137, 90.9137573, -179.2733459, 179.2527771
11: -85.6944580, 58.0569115, -85.6992798, 58.0681725, -143.7626343, 143.7561798
12: -97.5363922, 76.9938126, -97.5408936, 77.0135498, -174.5499115, 174.5346985
13: -84.9743195, 98.4069366, -84.9820404, 98.4134293, -183.3877563, 183.3889771
14: -144.1387329, 82.0367584, -144.1459656, 82.0530167, -226.1917419, 226.1827087
15: -78.6189194, 64.2293472, -78.6254730, 64.2419510, -142.8608704, 142.8548126
16: -90.9219894, 72.2189255, -90.9285889, 72.2316513, -163.1536407, 163.1475067
17: -133.2601013, 71.5994720, -133.2641754, 71.6139984, -204.8740997, 204.8636322
18: -93.2502136, 69.7798157, -93.2593613, 69.7869568, -163.0371704, 163.0391693
19: -67.6431503, 40.3744545, -67.6476440, 40.3813820, -108.0245285, 108.0220947
20: -68.4927826, 53.0498352, -68.4981995, 53.0517044, -121.5444870, 121.5480347
21: -85.0095367, 51.0976562, -85.0153656, 51.1057358, -136.1152649, 136.1130066
22: -86.4517441, 46.3523750, -86.4625778, 46.3626785, -132.8144226, 132.8149414
23: -69.9499817, 54.0042496, -69.9543610, 54.0099869, -123.9599686, 123.9586105
24: -90.2474670, 54.3928604, -90.2579498, 54.3971405, -144.6446075, 144.6508179
25: -76.0774384, 55.4253044, -76.0817947, 55.4296265, -131.5070496, 131.5070953
26: -100.8419876, 82.0008087, -100.8479691, 82.0224762, -182.8644714, 182.8487701
27: -87.7305756, 49.5499191, -87.7433319, 49.5531921, -137.2837677, 137.2932434
28: -68.4424133, 54.4309959, -68.4512634, 54.4341240, -122.8765411, 122.8822556
29: -89.0498962, 41.7086792, -89.0556030, 41.7177353, -130.7676392, 130.7642822
30: -88.2500000, 63.5830078, -88.2561035, 63.5882034, -151.8381958, 151.8391113
31: -91.4648285, 56.0099602, -91.4702835, 56.0180092, -147.4828339, 147.4802399
32: -89.9080353, 57.4292450, -89.9157181, 57.4315948, -147.3396301, 147.3449707
33: -126.5937729, 77.7999420, -126.6075821, 77.8037262, -204.3974915, 204.4075012
34: -106.2891083, 48.5881233, -106.3023224, 48.5918198, -154.8809204, 154.8904419
35: -99.0533829, 58.6301880, -99.0655060, 58.6322708, -157.6856537, 157.6956940
36: -92.4339600, 57.0902100, -92.4463806, 57.0916138, -149.5255737, 149.5365906
37: -145.2404938, 62.5749664, -145.2507629, 62.5788803, -207.8193665, 207.8257294
38: -112.1757736, 71.0741577, -112.1908798, 71.0778809, -183.2536316, 183.2650452
39: -133.0268860, 76.3446198, -133.0374908, 76.3484039, -209.3752899, 209.3821106
40: -110.8625259, 56.7115517, -110.8736649, 56.7135658, -167.5760956, 167.5852051
41: -95.7368851, 65.7791443, -95.7469406, 65.7821350, -161.5190125, 161.5260925
42: -70.2154694, 56.5964203, -70.2232819, 56.6005936, -126.8160629, 126.8197021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1669
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
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1699
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
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1731
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
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 876
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
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 545
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
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1689
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
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 956
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
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 1684
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 934
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
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 851
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
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1786
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
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 537
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
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 984
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
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1249
type: B, layer: 1, pos: 1037
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1649
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.2520455, upper bound: 91.4569046
time: 292.97 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.2520455, upper bound: 91.4569046
time: 114.56 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -108.5506516, 78.7246170, -108.8890533, 78.9301682, -187.4808044, 187.6136780
1: -56.9843712, 58.9458046, -57.1861763, 59.1177788, -116.1021500, 116.1319809
2: -49.2399521, 60.3107491, -49.4411087, 60.5046082, -109.7445450, 109.7518616
3: -61.8565826, 73.2091599, -62.1207314, 73.5347290, -135.3913116, 135.3298950
4: -64.5778809, 70.4876480, -64.8147354, 70.7604980, -135.3383789, 135.3023682
5: -59.4774971, 72.8726349, -59.7395096, 73.1797943, -132.6572876, 132.6121368
6: -93.9823608, 62.7458153, -94.4048843, 62.9659157, -156.9482727, 157.1506958
7: -66.3727417, 69.3187027, -66.6842499, 69.5764008, -135.9491272, 136.0029449
8: -81.2441406, 83.6334381, -81.4485550, 83.8848114, -165.1289520, 165.0819855
9: -60.4303360, 76.3844223, -60.9880447, 76.9194336, -137.3497620, 137.3724670
10: -87.8765335, 90.4492493, -88.8867798, 91.3741302, -179.2506561, 179.3360291
11: -85.4578476, 57.9242706, -86.0606995, 58.2335052, -143.6913452, 143.9849701
12: -97.2437210, 76.7960815, -97.7283707, 77.1106339, -174.3543549, 174.5244446
13: -84.8233566, 98.2853851, -85.1184235, 98.5671082, -183.3904572, 183.4038086
14: -143.7672119, 81.7253189, -144.5211639, 82.3664474, -226.1336365, 226.2464905
15: -78.4989471, 64.0957718, -78.7280273, 64.3894196, -142.8883667, 142.8237915
16: -90.6548080, 72.0003891, -91.3528214, 72.5534592, -163.2082672, 163.3532104
17: -133.0215454, 71.4710007, -133.6306763, 71.8080750, -204.8296204, 205.1016388
18: -93.1057663, 69.6566620, -93.4295120, 69.8980179, -163.0037842, 163.0861816
19: -67.4990768, 40.2989426, -67.8092651, 40.4296722, -107.9287415, 108.1082077
20: -68.3830109, 52.9986687, -68.6377945, 53.1353989, -121.5184097, 121.6364594
21: -84.8121872, 50.9863510, -85.2632217, 51.1973114, -136.0094910, 136.2495728
22: -86.3031616, 46.2641792, -86.7350922, 46.5111580, -132.8143158, 132.9992676
23: -69.8331451, 53.9347229, -70.1171570, 54.0833893, -123.9165268, 124.0518723
24: -90.1385345, 54.3005524, -90.4396744, 54.5028572, -144.6413879, 144.7402344
25: -75.9751282, 55.3318787, -76.2118378, 55.5473480, -131.5224762, 131.5437012
26: -100.6410675, 81.8202820, -101.0291595, 82.1148758, -182.7559509, 182.8494415
27: -87.5329666, 49.4462128, -87.9339600, 49.6478577, -137.1808167, 137.3801727
28: -68.3119354, 54.3198624, -68.5447540, 54.5071869, -122.8191223, 122.8646164
29: -88.9080734, 41.6175652, -89.2732849, 41.7841644, -130.6922302, 130.8908539
30: -88.1417923, 63.4941139, -88.4280319, 63.7493286, -151.8911133, 151.9221497
31: -91.3166199, 55.9384460, -91.6356583, 56.0657654, -147.3823853, 147.5740967
32: -89.7597885, 57.3673592, -90.0989990, 57.5188370, -147.2786255, 147.4663544
33: -126.3591843, 77.5131989, -126.9462662, 78.1694412, -204.5285950, 204.4594727
34: -106.0513229, 48.3582001, -106.4589767, 48.8153572, -154.8666534, 154.8171692
35: -98.8432693, 58.3575058, -99.2833557, 58.9510498, -157.7943115, 157.6408691
36: -92.2039261, 56.9101715, -92.6912460, 57.3675194, -149.5714417, 149.6014099
37: -145.0395966, 62.4041977, -145.7216492, 62.8991470, -207.9387207, 208.1258545
38: -111.8955765, 70.8134384, -112.5049591, 71.4472046, -183.3427734, 183.3183899
39: -132.8282166, 76.1567230, -133.4562378, 76.7051620, -209.5333862, 209.6129456
40: -110.6603088, 56.5544662, -111.1867905, 56.9553337, -167.6156311, 167.7412567
41: -95.5523834, 65.6622238, -96.0008774, 65.9393005, -161.4916840, 161.6631012
42: -70.0766296, 56.5032234, -70.4057312, 56.6734161, -126.7500458, 126.9089432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1686
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
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1414
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 656
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
type: B, layer: 1, pos: 1760
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.2520455, upper bound: 91.2918857
time: 109.41 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.2520455, upper bound: 91.2918857
time: 111.23 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -108.6696320, 78.7972336, -108.8994217, 78.9371490, -187.6067810, 187.6966553
1: -57.0531540, 59.0252686, -57.1914215, 59.1242180, -116.1773605, 116.2166901
2: -49.3495026, 60.4134102, -49.4584503, 60.5118599, -109.8613586, 109.8718414
3: -62.0051155, 73.3640747, -62.1442108, 73.5433044, -135.5484009, 135.5082855
4: -64.6748352, 70.6014099, -64.8275452, 70.7687378, -135.4435730, 135.4289551
5: -59.5821762, 72.9992371, -59.7527466, 73.1882782, -132.7704468, 132.7519836
6: -94.2035522, 62.8539047, -94.4395905, 62.9712791, -157.1748352, 157.2934570
7: -66.5003357, 69.4254608, -66.6980133, 69.5838928, -136.0842133, 136.1234589
8: -81.3386002, 83.7470093, -81.4588928, 83.8931885, -165.2317810, 165.2059021
9: -60.6712837, 76.6020432, -60.9967117, 76.9604874, -137.6317749, 137.5987549
10: -88.3595886, 90.8887634, -88.8980865, 91.4584351, -179.8180237, 179.7868500
11: -85.6944580, 58.0569115, -86.0699158, 58.2568703, -143.9513245, 144.1268311
12: -97.5363922, 76.9938126, -97.7399750, 77.1444550, -174.6808472, 174.7337952
13: -84.9743195, 98.4069366, -85.1313934, 98.5798340, -183.5541534, 183.5383301
14: -144.1387329, 82.0367584, -144.5363464, 82.4302673, -226.5690002, 226.5731049
15: -78.6189194, 64.2293472, -78.7393494, 64.4055023, -143.0243988, 142.9686890
16: -90.9219894, 72.2189255, -91.3663330, 72.5916443, -163.5136414, 163.5852661
17: -133.2601013, 71.5994720, -133.6407623, 71.8252716, -205.0853729, 205.2402344
18: -93.2502136, 69.7798157, -93.4409256, 69.9117737, -163.1619873, 163.2207336
19: -67.6431503, 40.3744545, -67.8207245, 40.4390869, -108.0822296, 108.1951752
20: -68.4927826, 53.0498352, -68.6473236, 53.1399155, -121.6326981, 121.6971588
21: -85.0095367, 51.0976562, -85.2757263, 51.2143250, -136.2238617, 136.3733826
22: -86.4517441, 46.3523750, -86.7524490, 46.5206375, -132.9723816, 133.1048126
23: -69.9499817, 54.0042496, -70.1259613, 54.0919952, -124.0419769, 124.1302109
24: -90.2474670, 54.3928604, -90.4561539, 54.5083275, -144.7557983, 144.8489990
25: -76.0774384, 55.4253044, -76.2210464, 55.5559006, -131.6333313, 131.6463470
26: -100.8419876, 82.0008087, -101.0408936, 82.1430206, -182.9849854, 183.0416870
27: -87.7305756, 49.5499191, -87.9674759, 49.6528435, -137.3834229, 137.5173950
28: -68.4424133, 54.4309959, -68.5641937, 54.5132751, -122.9556885, 122.9951706
29: -89.0498962, 41.7086792, -89.2886429, 41.7980881, -130.8479919, 130.9973145
30: -88.2500000, 63.5830078, -88.4379654, 63.7602844, -152.0102844, 152.0209656
31: -91.4648285, 56.0099602, -91.6493073, 56.0744362, -147.5392609, 147.6592560
32: -89.9080353, 57.4292450, -90.1197739, 57.5225296, -147.4305573, 147.5490112
33: -126.5937729, 77.7999420, -126.9901199, 78.1772766, -204.7710571, 204.7900543
34: -106.2891083, 48.5881233, -106.5020218, 48.8225098, -155.1116028, 155.0901337
35: -99.0533829, 58.6301880, -99.3242645, 58.9567566, -158.0101318, 157.9544373
36: -92.4339600, 57.0902100, -92.7307129, 57.3712730, -149.8052368, 149.8209229
37: -145.2404938, 62.5749664, -145.7503052, 62.9071617, -208.1476593, 208.3252716
38: -112.1757736, 71.0741577, -112.5527420, 71.4552002, -183.6309509, 183.6268921
39: -133.0268860, 76.3446198, -133.4844666, 76.7112885, -209.7381744, 209.8290863
40: -110.8625259, 56.7115517, -111.2193756, 56.9590912, -167.8216248, 167.9309082
41: -95.7368851, 65.7791443, -96.0307007, 65.9445877, -161.6814728, 161.8098450
42: -70.2154694, 56.5964203, -70.4217682, 56.6814651, -126.8969345, 127.0181885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=520, inp2_unstable=521, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=638, inp2_unstable=638, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=29, inp2_unstable=29, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1669
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.2520455, upper bound: 91.3609827
time: 172.83 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -91.2520455, upper bound: 91.3609827
time: 641.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 817.11 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 817.11
Output dim: 9, lower bound: -91.2520455, upper bound: 91.3467921
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 817.11
Output dim: 9, lower bound: -91.2520455, upper bound: 91.3467921
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 817.11
Output dim: 9, lower bound: -91.2520455, upper bound: 91.4569046
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 817.11
Output dim: 9, lower bound: -91.2520455, upper bound: 91.4569046
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 817.11
Output dim: 9, lower bound: -91.2520455, upper bound: 91.2918857
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 817.11
Output dim: 9, lower bound: -91.2520455, upper bound: 91.2918857
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 817.11
Output dim: 9, lower bound: -91.2520455, upper bound: 91.3609827
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 817.11
Output dim: 9, lower bound: -91.2520455, upper bound: 91.3609827
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 817.11
Output dim: 9, lower bound: -91.3680006, upper bound: 91.4294947
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 817.11
Output dim: 9, lower bound: -91.3680006, upper bound: 91.5558787
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 817.11
Output dim: 9, lower bound: -91.3680006, upper bound: 91.4294950
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 817.11
Output dim: 9, lower bound: -91.3680006, upper bound: 91.5558791
Binary search (step 1): status=Status.UNKNOWN, k_low=8, k_high=9, k_mid=8, eps_mid=0.0312500, abs_max=137.98736572265625
rel_dist={9: [-91.57507985954415, 91.57507985872158]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.02734375
execution time: 8304.42 seconds
