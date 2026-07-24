## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 7200 seconds
Split limit: 100
Threshold: 70.1762913496


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=457, inp2_unstable=457, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-91.4896622, 65.8297272, -91.4896622, 65.8297272, -157.3193970, 157.3193817)
1: (-45.6620483, 56.0444260, -45.6620483, 56.0444260, -101.7064743, 101.7064743)
2: (-40.0057907, 57.0831985, -40.0057907, 57.0831985, -97.0889816, 97.0889816)
3: (-50.0993347, 59.1422462, -50.0993347, 59.1422462, -109.2415771, 109.2415771)
4: (-48.9216766, 73.0156403, -48.9216766, 73.0156403, -121.9372940, 121.9373169)
5: (-46.3402328, 58.0639458, -46.3402328, 58.0639458, -104.4041748, 104.4041748)
6: (-90.9129562, 43.8431740, -90.9129562, 43.8431740, -134.7561188, 134.7561340)
7: (-54.9336319, 56.6945648, -54.9336319, 56.6945648, -111.6281967, 111.6281967)
8: (-60.8093033, 82.6538086, -60.8093033, 82.6538086, -143.4631042, 143.4630890)
9: (-49.4784508, 63.6636848, -49.4784508, 63.6636848, -113.1421356, 113.1421356)
10: (-76.5874405, 72.1994781, -76.5874405, 72.1994781, -148.7869263, 148.7869263)
11: (-80.6512146, 37.6950340, -80.6512146, 37.6950340, -118.3462372, 118.3462448)
12: (-84.7720871, 51.4539680, -84.7720871, 51.4539680, -136.2260437, 136.2260437)
13: (-77.5684052, 80.7841492, -77.5684052, 80.7841492, -158.3525543, 158.3525543)
14: (-117.3127670, 55.8701401, -117.3127670, 55.8701401, -173.1828918, 173.1828918)
15: (-60.6139526, 63.1955643, -60.6139526, 63.1955643, -123.8095093, 123.8095169)
16: (-79.1500015, 54.9147339, -79.1500015, 54.9147339, -134.0647278, 134.0647278)
17: (-110.6607285, 47.8645630, -110.6607285, 47.8645630, -158.5252686, 158.5252991)
18: (-78.8685837, 54.3912735, -78.8685837, 54.3912735, -133.2598572, 133.2598572)
19: (-57.6920547, 36.1438560, -57.6920547, 36.1438560, -93.8359070, 93.8359070)
20: (-56.4062538, 39.8666687, -56.4062538, 39.8666687, -96.2729187, 96.2729187)
21: (-73.9520569, 41.6378708, -73.9520569, 41.6378708, -115.5899277, 115.5899277)
22: (-69.0435486, 44.0636444, -69.0435486, 44.0636444, -113.1071930, 113.1071930)
23: (-61.4972343, 46.7317924, -61.4972343, 46.7317924, -108.2290192, 108.2290192)
24: (-73.4364166, 46.1977005, -73.4364166, 46.1977005, -119.6341095, 119.6341095)
25: (-64.1333618, 47.5422401, -64.1333618, 47.5422401, -111.6755981, 111.6755981)
26: (-82.8443909, 61.9199905, -82.8443909, 61.9199905, -144.7643738, 144.7643738)
27: (-69.3682251, 45.9550247, -69.3682251, 45.9550247, -115.3232422, 115.3232422)
28: (-58.3477058, 48.8492203, -58.3477058, 48.8492203, -107.1969299, 107.1969223)
29: (-75.0632477, 42.2576370, -75.0632477, 42.2576370, -117.3208847, 117.3208847)
30: (-78.9728622, 47.9274483, -78.9728622, 47.9274483, -126.9002838, 126.9002838)
31: (-80.0942993, 47.9363213, -80.0942993, 47.9363213, -128.0306244, 128.0306244)
32: (-83.4712524, 42.7773972, -83.4712524, 42.7773972, -126.2486420, 126.2486496)
33: (-109.8924103, 52.0851135, -109.8924103, 52.0851135, -161.9775085, 161.9775238)
34: (-97.8313904, 28.4801559, -97.8313904, 28.4801559, -126.3115463, 126.3115387)
35: (-91.5555267, 39.6454468, -91.5555267, 39.6454468, -131.2009583, 131.2009583)
36: (-90.0425873, 45.5510864, -90.0425873, 45.5510864, -135.5936584, 135.5936737)
37: (-131.4534607, 40.4228439, -131.4534607, 40.4228439, -171.8763123, 171.8763123)
38: (-106.7509155, 49.6612320, -106.7509155, 49.6612320, -156.4121399, 156.4121399)
39: (-118.5973587, 57.2216721, -118.5973587, 57.2216721, -175.8190308, 175.8190308)
40: (-100.1530228, 35.2989044, -100.1530228, 35.2989044, -135.4519348, 135.4519348)
41: (-84.2078705, 51.1618767, -84.2078705, 51.1618767, -135.3697205, 135.3697357)
42: (-66.2317963, 38.1462288, -66.2317963, 38.1462288, -104.3780212, 104.3780212)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.90 + 93.09 = 95.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -70.3169252, upper bound: 70.3169252

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1671

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2708018, upper bound: 70.3138344
time: 76.12 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2708018, upper bound: 70.3138344
time: 123.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 200.13 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 200.13
Output dim: 4, lower bound: -70.2708018, upper bound: 70.3138344
IS_A2, status: Status.UNKNOWN, split count: 1, time: 200.13
Output dim: 4, lower bound: -70.2708018, upper bound: 70.3138344

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -91.3213043, 65.7857513, -91.3992386, 65.8063202, -157.1276093, 157.1849823
1: -45.5418701, 56.0182419, -45.5978699, 56.0304985, -101.5723572, 101.6161118
2: -39.8204613, 57.0575562, -39.9070244, 57.0695724, -96.8900299, 96.9645844
3: -49.8970680, 59.1013107, -49.9915657, 59.1204834, -109.0175476, 109.0928726
4: -48.7142906, 72.9826508, -48.8110619, 72.9981155, -121.7123871, 121.7937012
5: -46.1217842, 58.0246353, -46.2207794, 58.0430565, -104.1648407, 104.2454071
6: -90.8448639, 43.7711296, -90.8766937, 43.8047104, -134.6495667, 134.6478271
7: -54.7714386, 56.6660614, -54.8459129, 56.6794128, -111.4508514, 111.5119705
8: -60.6149902, 82.6140213, -60.7056313, 82.6326752, -143.2476501, 143.3196411
9: -49.4227333, 63.5123978, -49.4487610, 63.5828514, -113.0055847, 112.9611588
10: -76.5075378, 71.8824463, -76.5449066, 72.0306396, -148.5381775, 148.4273529
11: -80.5862885, 37.4702682, -80.6167297, 37.5753937, -118.1616745, 118.0869980
12: -84.7255707, 51.1341171, -84.7473679, 51.2826538, -136.0082245, 135.8814850
13: -77.4858856, 80.6926575, -77.5244370, 80.7356644, -158.2215271, 158.2171021
14: -117.2267990, 55.5879097, -117.2670059, 55.7199783, -172.9467773, 172.8549194
15: -60.4579048, 63.1360359, -60.5294266, 63.1639481, -123.6218567, 123.6654663
16: -79.0574265, 54.7088737, -79.1008224, 54.8050919, -133.8625183, 133.8096924
17: -110.6097488, 47.7281609, -110.6335602, 47.7919540, -158.4016724, 158.3617249
18: -78.8070526, 54.2625046, -78.8358154, 54.3226967, -133.1297455, 133.0983276
19: -57.6342239, 36.0015602, -57.6612854, 36.0680389, -93.7022629, 93.6628418
20: -56.3461571, 39.7109489, -56.3743439, 39.7837753, -96.1299286, 96.0852814
21: -73.8945923, 41.4410095, -73.9215088, 41.5330429, -115.4276352, 115.3625183
22: -68.9924240, 43.9675217, -69.0163727, 44.0124130, -113.0048218, 112.9838943
23: -61.4508209, 46.6096725, -61.4725914, 46.6667404, -108.1175385, 108.0822525
24: -73.3691940, 46.1511002, -73.4006195, 46.1725616, -119.5417557, 119.5517197
25: -64.0892487, 47.4152641, -64.1099167, 47.4746017, -111.5638504, 111.5251770
26: -82.7797394, 61.6532631, -82.8100433, 61.7761269, -144.5558624, 144.4633026
27: -69.2690811, 45.9126091, -69.3150330, 45.9324036, -115.2014771, 115.2276306
28: -58.2973595, 48.7280655, -58.3209534, 48.7835503, -107.0809097, 107.0490189
29: -75.0226898, 42.1504745, -75.0416870, 42.1998520, -117.2225418, 117.1921387
30: -78.9212570, 47.7777481, -78.9454193, 47.8477135, -126.7689667, 126.7231674
31: -80.0189209, 47.7652168, -80.0542450, 47.8452034, -127.8641205, 127.8194580
32: -83.4178314, 42.6529427, -83.4428253, 42.7110634, -126.1288910, 126.0957489
33: -109.7412338, 52.0245743, -109.8119583, 52.0528755, -161.7940979, 161.8365326
34: -97.7315979, 28.4245987, -97.7782288, 28.4505253, -126.1821213, 126.2028122
35: -91.4567719, 39.5986328, -91.5030212, 39.6205368, -131.0773010, 131.1016388
36: -89.9831314, 45.4956589, -90.0109177, 45.5213814, -135.5045166, 135.5065765
37: -131.3728943, 40.3259125, -131.4104919, 40.3708038, -171.7436981, 171.7363892
38: -106.6347885, 49.6071091, -106.6887741, 49.6323051, -156.2670898, 156.2958679
39: -118.4963913, 57.1340942, -118.5435104, 57.1745262, -175.6709137, 175.6775818
40: -100.0724792, 35.2577820, -100.1101837, 35.2768440, -135.3493195, 135.3679504
41: -84.1393890, 51.1012497, -84.1714020, 51.1291199, -135.2685089, 135.2726440
42: -66.1784515, 38.0395699, -66.2034149, 38.0893669, -104.2678223, 104.2429810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=457, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2550695, upper bound: 70.2810698
time: 114.95 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2550695, upper bound: 70.3030013
time: 90.65 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -91.5572815, 65.9842834, -91.4688568, 65.8242950, -157.3815765, 157.4531403
1: -45.7041969, 56.2767944, -45.6491737, 56.0409241, -101.7451172, 101.9259644
2: -40.0440826, 57.3512192, -39.9891167, 57.0786133, -97.1226959, 97.3403320
3: -50.1253052, 59.4354057, -50.0784531, 59.1357574, -109.2610626, 109.5138550
4: -48.9753952, 73.3160248, -48.9032555, 73.0089264, -121.9843140, 122.2192764
5: -46.3873367, 58.3225212, -46.3241196, 58.0578880, -104.4452209, 104.6466370
6: -91.0342331, 43.8610458, -90.9026337, 43.8060532, -134.8402710, 134.7636719
7: -55.0119286, 56.8803864, -54.9157181, 56.6897507, -111.7016754, 111.7961044
8: -60.8534088, 82.9740524, -60.7908211, 82.6482086, -143.5016174, 143.7648621
9: -49.6685791, 63.7269135, -49.4695282, 63.6499329, -113.3185043, 113.1964417
10: -77.0238342, 72.2536621, -76.5766449, 72.1702576, -149.1940918, 148.8303070
11: -81.0911407, 37.7204285, -80.6430740, 37.6726074, -118.7637329, 118.3635025
12: -85.1889343, 51.4995117, -84.7649765, 51.4259491, -136.6148682, 136.2644958
13: -77.5886230, 80.8915405, -77.5383453, 80.7734909, -158.3621216, 158.4298859
14: -117.6962051, 55.9012146, -117.3011627, 55.8451004, -173.5413055, 173.2023773
15: -60.6601868, 63.3568382, -60.5722237, 63.1884193, -123.8485947, 123.9290619
16: -79.4302521, 54.9584122, -79.1355743, 54.8833237, -134.3135681, 134.0939941
17: -110.9078140, 47.9288902, -110.6536865, 47.8467865, -158.7546082, 158.5825806
18: -79.0908661, 54.4297676, -78.8612289, 54.3763924, -133.4672546, 133.2910004
19: -57.9276581, 36.1629868, -57.6850777, 36.1285553, -94.0562134, 93.8480682
20: -56.6052322, 39.8797951, -56.3994102, 39.8501358, -96.4553528, 96.2792053
21: -74.3057709, 41.6711502, -73.9436035, 41.6195831, -115.9253464, 115.6147537
22: -69.1615906, 44.1064987, -69.0357208, 44.0378914, -113.1994781, 113.1422195
23: -61.6743546, 46.7657280, -61.4908943, 46.7199783, -108.3943176, 108.2566223
24: -73.4954834, 46.2025146, -73.4199371, 46.1793137, -119.6747971, 119.6224518
25: -64.2450790, 47.5862198, -64.1257858, 47.5292282, -111.7743073, 111.7120056
26: -83.1230164, 61.9699249, -82.8342896, 61.8941269, -145.0171509, 144.8042145
27: -69.4592133, 45.9666443, -69.3555374, 45.9420395, -115.4012451, 115.3221817
28: -58.4603424, 48.8746986, -58.3415756, 48.8361435, -107.2964783, 107.2162628
29: -75.2252655, 42.2832603, -75.0554733, 42.2362137, -117.4614716, 117.3387299
30: -79.1736145, 47.9719620, -78.9644928, 47.9121017, -127.0857162, 126.9364548
31: -80.3591156, 47.9559555, -80.0862198, 47.9172325, -128.2763519, 128.0421753
32: -83.6677551, 42.8102150, -83.4630432, 42.7638626, -126.4316101, 126.2732544
33: -109.9391403, 52.2600403, -109.8753510, 52.0781212, -162.0172577, 162.1353760
34: -97.8776474, 28.5842056, -97.8193436, 28.4730110, -126.3506622, 126.4035492
35: -91.5763474, 39.7871742, -91.5400238, 39.6412468, -131.2175903, 131.3271942
36: -90.1005859, 45.6061096, -90.0296936, 45.5376396, -135.6382294, 135.6358032
37: -131.5559387, 40.4748306, -131.4369812, 40.4053650, -171.9613037, 171.9118042
38: -106.8374786, 49.7533455, -106.7328033, 49.6519661, -156.4894409, 156.4861450
39: -118.6868591, 57.3069305, -118.5820160, 57.2095642, -175.8964233, 175.8889465
40: -100.2728806, 35.3652458, -100.1420212, 35.2840843, -135.5569611, 135.5072632
41: -84.2885742, 51.1965332, -84.1977692, 51.1429367, -135.4315033, 135.3942871
42: -66.3551025, 38.1627693, -66.2230377, 38.0950775, -104.4501648, 104.3857956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=457, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2550695, upper bound: 70.2810698
time: 165.64 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2550695, upper bound: 70.3030013
time: 84.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 252.42 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 252.42
Output dim: 4, lower bound: -70.2550695, upper bound: 70.2810698
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 252.42
Output dim: 4, lower bound: -70.2550695, upper bound: 70.3030013
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 252.42
Output dim: 4, lower bound: -70.2550695, upper bound: 70.2810698
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 252.42
Output dim: 4, lower bound: -70.2550695, upper bound: 70.3030013

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -91.2676544, 65.7573700, -91.2868881, 65.7463226, -157.0139771, 157.0442505
1: -45.5119057, 56.0065002, -45.5349884, 56.0057678, -101.5176697, 101.5414810
2: -39.7694321, 57.0451851, -39.7997017, 57.0435333, -96.8129578, 96.8448868
3: -49.8392487, 59.0799446, -49.8707275, 59.0754776, -108.9147186, 108.9506683
4: -48.6478577, 72.9620285, -48.6715202, 72.9545898, -121.6024399, 121.6335449
5: -46.0640259, 58.0015869, -46.0978966, 57.9943695, -104.0583954, 104.0994873
6: -90.8072357, 43.7264481, -90.7971497, 43.7126999, -134.5199280, 134.5235901
7: -54.7358780, 56.6461945, -54.7707443, 56.6374359, -111.3733139, 111.4169388
8: -60.5470505, 82.5912781, -60.5626068, 82.5847473, -143.1318054, 143.1538849
9: -49.4019699, 63.4347610, -49.4051323, 63.4214172, -112.8233871, 112.8398895
10: -76.4741669, 71.7277145, -76.4746704, 71.7043610, -148.1785126, 148.2023773
11: -80.5496140, 37.3670044, -80.5391769, 37.3571625, -117.9067688, 117.9061737
12: -84.7009125, 51.0036354, -84.6954803, 51.0062408, -135.7071533, 135.6991119
13: -77.4632721, 80.6170731, -77.4767303, 80.5757141, -158.0389862, 158.0937958
14: -117.1888199, 55.4415512, -117.1873932, 55.4109268, -172.5997467, 172.6289368
15: -60.3758392, 63.1060982, -60.3561783, 63.1007500, -123.4765778, 123.4622726
16: -79.0178146, 54.6069336, -79.0175400, 54.5906906, -133.6085052, 133.6244812
17: -110.5863495, 47.6560631, -110.5848312, 47.6414871, -158.2278442, 158.2408905
18: -78.7701569, 54.2156029, -78.7582550, 54.2235260, -132.9936829, 132.9738617
19: -57.6071815, 35.9627304, -57.6043739, 35.9869041, -93.5940704, 93.5671005
20: -56.3183212, 39.6499443, -56.3155251, 39.6546822, -95.9729919, 95.9654694
21: -73.8653183, 41.3764191, -73.8597870, 41.3961868, -115.2615051, 115.2361984
22: -68.9285965, 43.9372215, -68.8821640, 43.9482994, -112.8768921, 112.8193741
23: -61.4296455, 46.5758514, -61.4279785, 46.5965691, -108.0262146, 108.0038223
24: -73.3034286, 46.1386490, -73.2621155, 46.1461258, -119.4495544, 119.4007645
25: -64.0581360, 47.3761711, -64.0444183, 47.3923302, -111.4504700, 111.4205933
26: -82.7452698, 61.5706215, -82.7374115, 61.6010170, -144.3462830, 144.3080139
27: -69.1838150, 45.8988152, -69.1349335, 45.9031334, -115.0869446, 115.0337524
28: -58.2555161, 48.7076454, -58.2322350, 48.7405090, -106.9960175, 106.9398727
29: -74.9789276, 42.1250305, -74.9497070, 42.1460228, -117.1249542, 117.0747375
30: -78.8904419, 47.7336502, -78.8803711, 47.7562332, -126.6466675, 126.6140137
31: -79.9840088, 47.7012253, -79.9807129, 47.7096634, -127.6936646, 127.6819382
32: -83.3831329, 42.5920677, -83.3697205, 42.5828972, -125.9660339, 125.9617920
33: -109.6509094, 51.9973679, -109.6204681, 51.9955254, -161.6464386, 161.6178284
34: -97.6683197, 28.4027367, -97.6443939, 28.4046516, -126.0729675, 126.0471344
35: -91.3868256, 39.5764389, -91.3549271, 39.5737991, -130.9606323, 130.9313660
36: -89.9419708, 45.4752541, -89.9238052, 45.4777756, -135.4197388, 135.3990479
37: -131.2992706, 40.3031616, -131.2543945, 40.3229065, -171.6221619, 171.5575562
38: -106.5910263, 49.5549583, -106.5964890, 49.5239944, -156.1150208, 156.1514435
39: -118.4408646, 57.0893936, -118.4262619, 57.0802155, -175.5210876, 175.5156555
40: -100.0247650, 35.2351532, -100.0095367, 35.2297668, -135.2545319, 135.2446899
41: -84.1007996, 51.0836029, -84.0900116, 51.0927162, -135.1935120, 135.1735992
42: -66.1461716, 37.9848900, -66.1352005, 37.9751625, -104.1213379, 104.1200867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2438475
time: 79.15 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2436978
time: 73.65 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -91.2945328, 65.7564697, -91.5116272, 65.8475647, -157.1420898, 157.2680969
1: -45.5275230, 56.0137444, -45.6764565, 56.1578331, -101.6853561, 101.6902008
2: -39.7992325, 57.0490570, -39.9633484, 57.2476120, -97.0468445, 97.0124054
3: -49.8599472, 59.0895500, -50.0053635, 59.2723160, -109.1322632, 109.0949097
4: -48.6682854, 72.9685364, -48.8529053, 73.1939392, -121.8622284, 121.8214417
5: -46.0846710, 58.0143585, -46.2535400, 58.1641006, -104.2487640, 104.2678986
6: -90.8217316, 43.7214737, -91.0565491, 43.8189430, -134.6406708, 134.7780151
7: -54.7488136, 56.6546478, -54.9676476, 56.7422523, -111.4910660, 111.6222992
8: -60.5887985, 82.6032791, -60.7695503, 82.8572845, -143.4460754, 143.3728333
9: -49.4104729, 63.4862137, -49.6924324, 63.6285324, -113.0390015, 113.1786423
10: -76.4927826, 71.8324356, -77.1492157, 72.0652771, -148.5580597, 148.9816589
11: -80.5717468, 37.4364128, -81.2071762, 37.5975800, -118.1693268, 118.6435852
12: -84.7115021, 51.0963593, -85.2135315, 51.3475342, -136.0590363, 136.3098907
13: -77.4680710, 80.6580582, -77.5979919, 80.8431549, -158.3112183, 158.2560425
14: -117.2026978, 55.5471573, -117.8199844, 55.7481422, -172.9508362, 173.3671417
15: -60.4315529, 63.1217690, -60.6282005, 63.3931847, -123.8247375, 123.7499695
16: -79.0344238, 54.6733170, -79.5290833, 54.8430405, -133.8774719, 134.2023926
17: -110.5943680, 47.6876640, -110.9008408, 47.8725204, -158.4668579, 158.5884857
18: -78.7788391, 54.2352638, -79.0339432, 54.3687401, -133.1475525, 133.2692108
19: -57.6194611, 35.9797974, -57.7947960, 36.0928802, -93.7123413, 93.7745972
20: -56.3361320, 39.6901932, -56.5928268, 39.8019714, -96.1380920, 96.2830200
21: -73.8787155, 41.4215088, -74.2632217, 41.5793152, -115.4580307, 115.6847305
22: -68.9459076, 43.9512062, -69.0991211, 44.1581192, -113.1040268, 113.0503235
23: -61.4403915, 46.5961075, -61.5940132, 46.7102776, -108.1506653, 108.1901093
24: -73.3434296, 46.1362495, -73.4726868, 46.1930923, -119.5365143, 119.6089325
25: -64.0695190, 47.3999596, -64.1787262, 47.5395393, -111.6090546, 111.5786896
26: -82.7635345, 61.6175003, -83.0162201, 61.8395844, -144.6031189, 144.6337280
27: -69.2408371, 45.9039459, -69.4137115, 45.9862823, -115.2271194, 115.3176498
28: -58.2710419, 48.7107735, -58.3700180, 48.8120003, -107.0830383, 107.0807800
29: -75.0002823, 42.1355553, -75.1532135, 42.2750053, -117.2752838, 117.2887726
30: -78.9081726, 47.7525864, -79.0488739, 47.8947525, -126.8029175, 126.8014603
31: -80.0046310, 47.7417679, -80.2600861, 47.8708344, -127.8754654, 128.0018616
32: -83.4001160, 42.6292381, -83.6800842, 42.7337341, -126.1338501, 126.3093185
33: -109.7110291, 52.0131302, -109.8778763, 52.3166161, -162.0276337, 161.8910065
34: -97.7091599, 28.4138165, -97.8392181, 28.6886024, -126.3977661, 126.2530289
35: -91.4323883, 39.5907860, -91.5499878, 39.8679504, -131.3003387, 131.1407776
36: -89.9583817, 45.4810944, -90.0807037, 45.5989532, -135.5573273, 135.5617981
37: -131.3330994, 40.3163986, -131.5125885, 40.4637222, -171.7967987, 171.8289795
38: -106.6063843, 49.5741310, -106.8241119, 49.7179527, -156.3243408, 156.3982391
39: -118.4575119, 57.1232147, -118.6510239, 57.2766037, -175.7341156, 175.7742310
40: -100.0508423, 35.2403984, -100.2598495, 35.3661804, -135.4170227, 135.5002441
41: -84.1135941, 51.0901756, -84.2573471, 51.1875114, -135.3010864, 135.3475037
42: -66.1604309, 37.9612656, -66.3970184, 38.0571480, -104.2175751, 104.3582764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 955

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2641821
time: 129.39 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2639218
time: 83.39 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -91.5038147, 65.9563904, -91.3563919, 65.7643127, -157.2681274, 157.3127747
1: -45.6740646, 56.2650757, -45.5862617, 56.0161896, -101.6902466, 101.8513336
2: -39.9930267, 57.3389664, -39.8818245, 57.0526390, -97.0456696, 97.2207947
3: -50.0664482, 59.4142761, -49.9565582, 59.0908241, -109.1572723, 109.3708344
4: -48.9089928, 73.2957153, -48.7637711, 72.9655151, -121.8745117, 122.0594864
5: -46.3272324, 58.2996674, -46.1970863, 58.0093269, -104.3365479, 104.4967499
6: -90.9965973, 43.8167343, -90.8233109, 43.7140007, -134.7105865, 134.6400452
7: -54.9752808, 56.8608704, -54.8391457, 56.6477051, -111.6229858, 111.7000122
8: -60.7853775, 82.9514618, -60.6477890, 82.6003571, -143.3857422, 143.5992432
9: -49.6481171, 63.6494026, -49.4260101, 63.4885139, -113.1366272, 113.0754089
10: -76.9906616, 72.0990601, -76.5065460, 71.8440857, -148.8347473, 148.6056061
11: -81.0546951, 37.6171951, -80.5656204, 37.4543304, -118.5090256, 118.1828156
12: -85.1645508, 51.3691216, -84.7131805, 51.1495895, -136.3141479, 136.0823059
13: -77.5660553, 80.8161621, -77.4904327, 80.6136703, -158.1797180, 158.3065796
14: -117.6587524, 55.7549133, -117.2215729, 55.5359497, -173.1947021, 172.9764862
15: -60.5778732, 63.3272171, -60.3985786, 63.1254730, -123.7033386, 123.7257996
16: -79.3910294, 54.8566437, -79.0525360, 54.6688728, -134.0599060, 133.9091492
17: -110.8846283, 47.8564186, -110.6050034, 47.6955452, -158.5801697, 158.4614258
18: -79.0545197, 54.3827972, -78.7834320, 54.2771225, -133.3316345, 133.1662292
19: -57.9010468, 36.1243668, -57.6281776, 36.0475006, -93.9485474, 93.7525406
20: -56.5775108, 39.8188095, -56.3406830, 39.7210312, -96.2985382, 96.1594925
21: -74.2766571, 41.6066895, -73.8819885, 41.4827652, -115.7594223, 115.4886780
22: -69.0977936, 44.0758743, -68.9013977, 43.9738655, -113.0716476, 112.9772720
23: -61.6536636, 46.7319183, -61.4463272, 46.6498451, -108.3035126, 108.1782455
24: -73.4298553, 46.1895180, -73.2814484, 46.1525459, -119.5823975, 119.4709549
25: -64.2151031, 47.5470238, -64.0603180, 47.4468536, -111.6619568, 111.6073456
26: -83.0888672, 61.8854752, -82.7616119, 61.7188110, -144.8076782, 144.6470947
27: -69.3739700, 45.9527817, -69.1751709, 45.9127769, -115.2867432, 115.1279373
28: -58.4185905, 48.8539658, -58.2528839, 48.7928467, -107.2114410, 107.1068420
29: -75.1829681, 42.2573395, -74.9634094, 42.1823235, -117.3652954, 117.2207489
30: -79.1431351, 47.9279213, -78.8995438, 47.8206024, -126.9637375, 126.8274612
31: -80.3247681, 47.8919830, -80.0128479, 47.7816772, -128.1064453, 127.9048157
32: -83.6332855, 42.7495842, -83.3899918, 42.6357269, -126.2690125, 126.1395569
33: -109.8488312, 52.2332802, -109.6838379, 52.0207901, -161.8695984, 161.9171143
34: -97.8143768, 28.5625114, -97.6853180, 28.4270668, -126.2414398, 126.2478333
35: -91.5063095, 39.7651062, -91.3917618, 39.5946198, -131.1009216, 131.1568604
36: -90.0594025, 45.5847588, -89.9425049, 45.4930496, -135.5524597, 135.5272522
37: -131.4822388, 40.4518509, -131.2808533, 40.3573380, -171.8395691, 171.7326965
38: -106.7935028, 49.7014046, -106.6401901, 49.5431938, -156.3367004, 156.3415985
39: -118.6302567, 57.2615242, -118.4628067, 57.1135902, -175.7438507, 175.7243347
40: -100.2255630, 35.3423843, -100.0415726, 35.2363358, -135.4618988, 135.3839569
41: -84.2500992, 51.1790352, -84.1164017, 51.1063385, -135.3564453, 135.2954407
42: -66.3228760, 38.1083374, -66.1550369, 37.9807625, -104.3036346, 104.2633743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2438475
time: 76.44 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2436978
time: 85.52 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -91.5304794, 65.9539261, -91.5808792, 65.8658600, -157.3963318, 157.5348053
1: -45.6892281, 56.2724304, -45.7270622, 56.1683388, -101.8575668, 101.9994965
2: -40.0252914, 57.3426781, -40.0469208, 57.2565765, -97.2818680, 97.3895950
3: -50.0913925, 59.4237556, -50.0994072, 59.2876434, -109.3790054, 109.5231628
4: -48.9280548, 73.3020325, -48.9446793, 73.2052155, -122.1332703, 122.2467117
5: -46.3537292, 58.3121452, -46.3573608, 58.1788788, -104.5326080, 104.6695099
6: -91.0110626, 43.8105469, -91.0831451, 43.8218384, -134.8329010, 134.8936768
7: -54.9904327, 56.8686447, -55.0353661, 56.7527618, -111.7431946, 111.9040070
8: -60.8265190, 82.9633484, -60.8544540, 82.8725967, -143.6991119, 143.8178101
9: -49.6562119, 63.6998901, -49.7134399, 63.6951714, -113.3513794, 113.4133301
10: -77.0089264, 72.2024078, -77.1807480, 72.2035675, -149.2124786, 149.3831482
11: -81.0765839, 37.6861115, -81.2333450, 37.6948166, -118.7714005, 118.9194565
12: -85.1749878, 51.4610329, -85.2314301, 51.4904175, -136.6654053, 136.6924591
13: -77.5706329, 80.8559494, -77.6114731, 80.8825989, -158.4532318, 158.4674225
14: -117.6705780, 55.8604126, -117.8542404, 55.8736038, -173.5441742, 173.7146454
15: -60.6334877, 63.3426971, -60.6707764, 63.4172935, -124.0507812, 124.0134735
16: -79.4070740, 54.9217377, -79.5636826, 54.9210510, -134.3281250, 134.4854126
17: -110.8917236, 47.8882523, -110.9208908, 47.9278717, -158.8195953, 158.8091431
18: -79.0615082, 54.4042892, -79.0593567, 54.4245148, -133.4860229, 133.4636383
19: -57.9122849, 36.1403732, -57.8186455, 36.1531487, -94.0654297, 93.9590073
20: -56.5953445, 39.8585396, -56.6179810, 39.8677521, -96.4630890, 96.4765167
21: -74.2898102, 41.6509171, -74.2854309, 41.6653061, -115.9551163, 115.9363480
22: -69.1154480, 44.0897446, -69.1188889, 44.1839790, -113.2994232, 113.2086334
23: -61.6639900, 46.7514687, -61.6125755, 46.7631302, -108.4271164, 108.3640442
24: -73.4693375, 46.1883965, -73.4923630, 46.2010651, -119.6704025, 119.6807556
25: -64.2233734, 47.5707016, -64.1947784, 47.5941124, -111.8174896, 111.7654800
26: -83.1068268, 61.9389420, -83.0411835, 61.9646606, -145.0714722, 144.9801025
27: -69.4302063, 45.9578552, -69.4536057, 45.9957962, -115.4260025, 115.4114532
28: -58.4331551, 48.8575478, -58.3907928, 48.8645744, -107.2977295, 107.2483368
29: -75.2005920, 42.2679253, -75.1674576, 42.3101883, -117.5107803, 117.4353790
30: -79.1604462, 47.9458008, -79.0680542, 47.9594650, -127.1199112, 127.0138550
31: -80.3448944, 47.9320908, -80.2918625, 47.9429207, -128.2878113, 128.2239532
32: -83.6500244, 42.7855415, -83.7004395, 42.7860909, -126.4361115, 126.4859772
33: -109.9087524, 52.2473831, -109.9411469, 52.3417587, -162.2505035, 162.1885376
34: -97.8548889, 28.5724831, -97.8800201, 28.7108383, -126.5657196, 126.4524994
35: -91.5517349, 39.7785416, -91.5865936, 39.8881073, -131.4398499, 131.3651276
36: -90.0757675, 45.5923691, -90.0998993, 45.6161766, -135.6919250, 135.6922607
37: -131.5158691, 40.4647064, -131.5396729, 40.4969254, -172.0127869, 172.0043640
38: -106.8082352, 49.7203522, -106.8644409, 49.7381287, -156.5463562, 156.5847778
39: -118.6490784, 57.2965164, -118.6901703, 57.3120880, -175.9611664, 175.9866943
40: -100.2516785, 35.3476334, -100.2919235, 35.3740654, -135.6257324, 135.6395569
41: -84.2628632, 51.1841087, -84.2839203, 51.1996231, -135.4624939, 135.4680176
42: -66.3370514, 38.0899544, -66.4174652, 38.0728836, -104.4099350, 104.5074081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=456, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2641821
time: 77.73 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2436978
time: 89.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 169.68 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 169.68
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2438475
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 169.68
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2436978
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 169.68
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2641821
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 169.68
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2639218
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 169.68
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2438475
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 169.68
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2436978
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 169.68
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2641821
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 169.68
Output dim: 4, lower bound: -70.1856649, upper bound: 70.2436978

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -91.1885376, 65.7284088, -91.2722244, 65.7409668, -156.9295044, 157.0006409
1: -45.4486122, 55.9882050, -45.5232544, 56.0023613, -101.4509735, 101.5114517
2: -39.6823616, 57.0251083, -39.7835999, 57.0398331, -96.7221909, 96.8087082
3: -49.7301407, 59.0526733, -49.8505783, 59.0704689, -108.8005981, 108.9032516
4: -48.5426674, 72.9406433, -48.6520576, 72.9506683, -121.4933319, 121.5926971
5: -45.9655304, 57.9766579, -46.0796509, 57.9897461, -103.9552689, 104.0563049
6: -90.7702026, 43.6976128, -90.7903290, 43.7073669, -134.4775696, 134.4879303
7: -54.6636772, 56.6291122, -54.7571754, 56.6342697, -111.2979431, 111.3862915
8: -60.4642448, 82.5618973, -60.5472946, 82.5793304, -143.0435791, 143.1091919
9: -49.3655930, 63.3994675, -49.3983955, 63.4148140, -112.7804108, 112.7978668
10: -76.4313354, 71.5607681, -76.4667587, 71.6735382, -148.1048584, 148.0275116
11: -80.5098801, 37.2082291, -80.5317917, 37.3278885, -117.8377686, 117.7400208
12: -84.6691284, 50.8708839, -84.6895599, 50.9816971, -135.6508179, 135.5604401
13: -77.3542938, 80.5643387, -77.4559631, 80.5660400, -157.9203186, 158.0202942
14: -117.1281891, 55.2481384, -117.1761932, 55.3753166, -172.5035095, 172.4243317
15: -60.2839088, 63.0712433, -60.3391762, 63.0942993, -123.3781891, 123.4104156
16: -78.9559708, 54.5086021, -79.0060959, 54.5717163, -133.5276794, 133.5146942
17: -110.5447159, 47.5355263, -110.5770645, 47.6189499, -158.1636505, 158.1125946
18: -78.7227478, 54.0974731, -78.7495041, 54.2016869, -132.9244232, 132.8469849
19: -57.5713882, 35.8588104, -57.5977173, 35.9677353, -93.5391083, 93.4565277
20: -56.2814369, 39.5516663, -56.3087578, 39.6365509, -95.9179840, 95.8604126
21: -73.8271942, 41.2391319, -73.8527527, 41.3708839, -115.1980667, 115.0918884
22: -68.8969879, 43.8760262, -68.8762970, 43.9369812, -112.8339615, 112.7523117
23: -61.3997993, 46.4679298, -61.4224510, 46.5766678, -107.9764709, 107.8903809
24: -73.2690735, 46.0555687, -73.2557526, 46.1307755, -119.3998489, 119.3113098
25: -64.0266800, 47.2779388, -64.0385895, 47.3742027, -111.4008789, 111.3165283
26: -82.7037659, 61.4092674, -82.7297440, 61.5712204, -144.2749786, 144.1390076
27: -69.1411667, 45.8343430, -69.1270828, 45.8908844, -115.0320435, 114.9614182
28: -58.2218857, 48.6215515, -58.2260323, 48.7245789, -106.9464417, 106.8475800
29: -74.9505768, 42.0530586, -74.9444275, 42.1326637, -117.0832367, 116.9974670
30: -78.8550720, 47.6056900, -78.8738251, 47.7326126, -126.5876694, 126.4795151
31: -79.9362640, 47.5592880, -79.9718628, 47.6835327, -127.6197968, 127.5311356
32: -83.3416748, 42.5589828, -83.3620758, 42.5767365, -125.9184113, 125.9210510
33: -109.5512390, 51.9652557, -109.6020660, 51.9896240, -161.5408630, 161.5673218
34: -97.6024628, 28.3680305, -97.6321869, 28.3982430, -126.0007019, 126.0002136
35: -91.3210297, 39.5518341, -91.3424377, 39.5692596, -130.8902740, 130.8942719
36: -89.8762512, 45.4492493, -89.9117050, 45.4729996, -135.3492432, 135.3609619
37: -131.2464142, 40.2510071, -131.2445831, 40.3132286, -171.5596466, 171.4955902
38: -106.5133362, 49.5212212, -106.5820923, 49.5177689, -156.0310974, 156.1033173
39: -118.3576965, 57.0590057, -118.4103317, 57.0744553, -175.4321594, 175.4693298
40: -99.9756622, 35.2168655, -100.0004425, 35.2263489, -135.2020111, 135.2173004
41: -84.0454636, 51.0535088, -84.0796967, 51.0870972, -135.1325684, 135.1332092
42: -66.1185913, 37.9374962, -66.1300659, 37.9660797, -104.0846710, 104.0675659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1651

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1433764, upper bound: 70.2390084
time: 75.92 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1807179, upper bound: 70.2390084
time: 91.60 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -91.3942108, 65.8321533, -91.2501373, 65.7305603, -157.1247559, 157.0822906
1: -45.5873795, 56.2934189, -45.5124588, 55.9984322, -101.5858154, 101.8058777
2: -39.8293343, 57.3542175, -39.7766113, 57.0343399, -96.8636780, 97.1308289
3: -49.8769836, 59.5169487, -49.8428192, 59.0622749, -108.9392548, 109.3597717
4: -48.7349892, 73.3768845, -48.6445503, 72.9448700, -121.6798401, 122.0214386
5: -46.1266556, 58.3278542, -46.0726585, 57.9830856, -104.1097412, 104.4005051
6: -90.9446564, 43.7473755, -90.7795639, 43.6552391, -134.5998993, 134.5269470
7: -54.8552170, 56.8231239, -54.7486000, 56.6291199, -111.4843369, 111.5717239
8: -60.6173096, 82.9593964, -60.5403557, 82.5709381, -143.1882477, 143.4997559
9: -49.4845123, 63.5478363, -49.3833809, 63.4037514, -112.8882599, 112.9312134
10: -77.0511627, 71.8309250, -76.4595795, 71.6620178, -148.7131805, 148.2904968
11: -81.4243546, 37.3879166, -80.5181580, 37.3190346, -118.7433929, 117.9060745
12: -85.0871811, 51.0763626, -84.6831055, 50.9724884, -136.0596619, 135.7594452
13: -77.4581757, 81.0179596, -77.4112854, 80.5514832, -158.0096436, 158.4292297
14: -117.9034653, 55.4506836, -117.1629868, 55.3669853, -173.2704468, 172.6136780
15: -60.4064178, 63.4034767, -60.2838173, 63.0856438, -123.4920502, 123.6872940
16: -79.4814377, 54.6567726, -78.9877167, 54.5273438, -134.0087891, 133.6444702
17: -111.2573929, 47.7134247, -110.5687103, 47.6073456, -158.8647308, 158.2821350
18: -79.3236389, 54.2442169, -78.7372971, 54.1813202, -133.5049591, 132.9815063
19: -58.1283989, 35.9902573, -57.5912476, 35.9623642, -94.0907440, 93.5814972
20: -56.6588516, 39.6634293, -56.3030586, 39.6272964, -96.2861328, 95.9664764
21: -74.5616455, 41.4078598, -73.8453217, 41.3654633, -115.9271088, 115.2531815
22: -69.2012634, 43.9908371, -68.8702087, 43.9229889, -113.1242523, 112.8610458
23: -61.8918152, 46.6147690, -61.4162445, 46.5694427, -108.4612579, 108.0310059
24: -73.6069107, 46.1234283, -73.2411957, 46.1068344, -119.7137451, 119.3646240
25: -64.4050293, 47.4026794, -64.0277176, 47.3659058, -111.7709351, 111.4303970
26: -83.2436981, 61.6398277, -82.7187958, 61.5612297, -144.8049316, 144.3586121
27: -69.4185486, 45.9054604, -69.1156616, 45.8781357, -115.2966843, 115.0211182
28: -58.5378990, 48.7412796, -58.2210045, 48.7165298, -107.2544174, 106.9622726
29: -75.4087372, 42.1609459, -74.9373474, 42.1240845, -117.5328217, 117.0982819
30: -79.4097443, 47.7756653, -78.8605042, 47.7234879, -127.1332092, 126.6361542
31: -80.6265564, 47.7071609, -79.9632111, 47.6762924, -128.3028564, 127.6703720
32: -83.4700623, 42.6535110, -83.3407440, 42.5541916, -126.0242462, 125.9942474
33: -109.7266846, 52.3729095, -109.5886917, 51.9836273, -161.7102966, 161.9616089
34: -97.7330856, 28.5780182, -97.6221313, 28.3887215, -126.1218109, 126.2001419
35: -91.4090118, 39.8292427, -91.3201218, 39.5654221, -130.9744263, 131.1493683
36: -89.9883423, 45.6480370, -89.8910217, 45.4690208, -135.4573669, 135.5390625
37: -131.4450073, 40.3591919, -131.2206421, 40.2795830, -171.7245789, 171.5798340
38: -106.7130585, 49.7996368, -106.5674210, 49.5109940, -156.2240295, 156.3670654
39: -118.5485535, 57.2908096, -118.3888092, 57.0697861, -175.6183167, 175.6796112
40: -100.1714172, 35.4069939, -99.9864426, 35.2205811, -135.3919983, 135.3934326
41: -84.1874771, 51.1831322, -84.0624084, 51.0692177, -135.2566986, 135.2455444
42: -66.2715759, 38.0544662, -66.1246643, 37.9207878, -104.1923676, 104.1791306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1651

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1433764, upper bound: 70.2388843
time: 75.45 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1807179, upper bound: 70.2388843
time: 132.36 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -91.2153702, 65.7274704, -91.4969788, 65.8421936, -157.0575562, 157.2244568
1: -45.4641876, 55.9954414, -45.6647301, 56.1544647, -101.6186523, 101.6601715
2: -39.7121277, 57.0290375, -39.9472466, 57.2439346, -96.9560623, 96.9762802
3: -49.7508888, 59.0623322, -49.9852333, 59.2673645, -109.0182495, 109.0475616
4: -48.5629654, 72.9472504, -48.8334732, 73.1900787, -121.7530289, 121.7807083
5: -45.9861755, 57.9894180, -46.2353439, 58.1595116, -104.1456833, 104.2247467
6: -90.7847748, 43.6925812, -91.0497513, 43.8136063, -134.5983887, 134.7423401
7: -54.6767654, 56.6376610, -54.9541206, 56.7390747, -111.4158401, 111.5917816
8: -60.5060120, 82.5739136, -60.7542648, 82.8519135, -143.3579102, 143.3281555
9: -49.3740616, 63.4509315, -49.6857529, 63.6219559, -112.9960175, 113.1366882
10: -76.4499817, 71.6654739, -77.1413651, 72.0344086, -148.4843903, 148.8068390
11: -80.5320435, 37.2776260, -81.1998291, 37.5682907, -118.1003265, 118.4774551
12: -84.6797485, 50.9635963, -85.2076721, 51.3230019, -136.0027466, 136.1712646
13: -77.3590851, 80.6051559, -77.5772247, 80.8332520, -158.1923218, 158.1823730
14: -117.1421127, 55.3536835, -117.8088150, 55.7125168, -172.8546295, 173.1624756
15: -60.3396416, 63.0870018, -60.6112175, 63.3868713, -123.7265091, 123.6982193
16: -78.9726944, 54.5749664, -79.5178146, 54.8240814, -133.7967682, 134.0927734
17: -110.5527496, 47.5669174, -110.8931198, 47.8499756, -158.4027252, 158.4600220
18: -78.7315063, 54.1170578, -79.0251770, 54.3468437, -133.0783539, 133.1422424
19: -57.5837555, 35.8758545, -57.7881432, 36.0737343, -93.6574783, 93.6640015
20: -56.2992630, 39.5919418, -56.5860672, 39.7838135, -96.0830765, 96.1780090
21: -73.8406525, 41.2842712, -74.2561951, 41.5540390, -115.3946915, 115.5404663
22: -68.9142914, 43.8900070, -69.0931778, 44.1468773, -113.0611725, 112.9831848
23: -61.4105644, 46.4881859, -61.5884628, 46.6903763, -108.1009369, 108.0766449
24: -73.3090363, 46.0532112, -73.4662399, 46.1778107, -119.4868469, 119.5194397
25: -64.0380249, 47.3016968, -64.1728439, 47.5213890, -111.5594177, 111.4745331
26: -82.7221069, 61.4561348, -83.0085602, 61.8097839, -144.5318909, 144.4646912
27: -69.1982117, 45.8395309, -69.4057159, 45.9740677, -115.1722717, 115.2452469
28: -58.2373924, 48.6246643, -58.3637772, 48.7960854, -107.0334778, 106.9884338
29: -74.9719467, 42.0636406, -75.1479034, 42.2617531, -117.2336884, 117.2115402
30: -78.8728638, 47.6245499, -79.0422974, 47.8711624, -126.7440262, 126.6668396
31: -79.9568558, 47.5998611, -80.2512665, 47.8446655, -127.8015213, 127.8511200
32: -83.3586578, 42.5961151, -83.6724777, 42.7275467, -126.0861816, 126.2685852
33: -109.6113586, 51.9810677, -109.8595581, 52.3106842, -161.9220428, 161.8406067
34: -97.6432343, 28.3790512, -97.8270264, 28.6822128, -126.3254242, 126.2060699
35: -91.3665619, 39.5662689, -91.5374527, 39.8634949, -131.2300568, 131.1037292
36: -89.8926392, 45.4552078, -90.0685730, 45.5940475, -135.4866791, 135.5237732
37: -131.2801514, 40.2642326, -131.5028076, 40.4540329, -171.7341919, 171.7670288
38: -106.5287476, 49.5404167, -106.8097916, 49.7116356, -156.2403717, 156.3502045
39: -118.3740768, 57.0928497, -118.6351318, 57.2709732, -175.6450500, 175.7279663
40: -100.0019989, 35.2220840, -100.2508316, 35.3627357, -135.3647156, 135.4729156
41: -84.0582886, 51.0600662, -84.2470703, 51.1819458, -135.2402344, 135.3071289
42: -66.1329498, 37.9138527, -66.3919144, 38.0480270, -104.1809692, 104.3057709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1651

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1442196, upper bound: 70.2593461
time: 112.20 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1814746, upper bound: 70.2593461
time: 83.50 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -91.4207764, 65.8310928, -91.4747314, 65.8317719, -157.2525330, 157.3058167
1: -45.6033592, 56.3006516, -45.6538620, 56.1505737, -101.7539215, 101.9545135
2: -39.8596115, 57.3580704, -39.9402313, 57.2384644, -97.0980682, 97.2983017
3: -49.8980446, 59.5266838, -49.9774323, 59.2591934, -109.1572418, 109.5041199
4: -48.7552719, 73.3834076, -48.8258400, 73.1842804, -121.9395523, 122.2092438
5: -46.1478348, 58.3406944, -46.2283363, 58.1528244, -104.3006439, 104.5690308
6: -90.9594727, 43.7421570, -91.0390091, 43.7614746, -134.7209473, 134.7811584
7: -54.8693237, 56.8317642, -54.9454460, 56.7338791, -111.6031952, 111.7772064
8: -60.6595497, 82.9713287, -60.7472878, 82.8434906, -143.5030365, 143.7186127
9: -49.4932022, 63.5996704, -49.6707802, 63.6107368, -113.1039429, 113.2704468
10: -77.0697632, 71.9364014, -77.1342163, 72.0229111, -149.0926514, 149.0706177
11: -81.4464264, 37.4576302, -81.1862640, 37.5594330, -119.0058594, 118.6438904
12: -85.0977631, 51.1693420, -85.2012177, 51.3137207, -136.4114838, 136.3705444
13: -77.4630661, 81.0583344, -77.5324554, 80.8183441, -158.2814026, 158.5907898
14: -117.9174118, 55.5563393, -117.7955475, 55.7041245, -173.6215363, 173.3518829
15: -60.4625702, 63.4192543, -60.5555649, 63.3784409, -123.8410110, 123.9748230
16: -79.4979553, 54.7236328, -79.4998245, 54.7798004, -134.2777557, 134.2234497
17: -111.2655945, 47.7445145, -110.8846741, 47.8382187, -159.1037903, 158.6291809
18: -79.3322601, 54.2640381, -79.0128326, 54.3264427, -133.6586914, 133.2768707
19: -58.1411858, 36.0074921, -57.7816010, 36.0683823, -94.2095642, 93.7890930
20: -56.6766281, 39.7040176, -56.5803833, 39.7745743, -96.4511948, 96.2844009
21: -74.5751114, 41.4534302, -74.2487946, 41.5485992, -116.1237030, 115.7022171
22: -69.2186432, 44.0054359, -69.0870743, 44.1328812, -113.3515244, 113.0925140
23: -61.9025002, 46.6353531, -61.5822487, 46.6831818, -108.5856781, 108.2176056
24: -73.6467285, 46.1211662, -73.4515533, 46.1539307, -119.8006592, 119.5727234
25: -64.4162750, 47.4268494, -64.1618500, 47.5129929, -111.9292679, 111.5886993
26: -83.2620239, 61.6869431, -82.9975586, 61.7996979, -145.0617218, 144.6844788
27: -69.4753036, 45.9107208, -69.3941650, 45.9613457, -115.4366455, 115.3048859
28: -58.5532227, 48.7443085, -58.3587227, 48.7879791, -107.3412018, 107.1030273
29: -75.4301071, 42.1717796, -75.1408539, 42.2531891, -117.6832962, 117.3126297
30: -79.4274445, 47.7943192, -79.0289078, 47.8619576, -127.2893982, 126.8232193
31: -80.6470032, 47.7481728, -80.2426987, 47.8374405, -128.4844360, 127.9908752
32: -83.4871826, 42.6910400, -83.6511230, 42.7048111, -126.1919785, 126.3421631
33: -109.7868195, 52.3886681, -109.8460541, 52.3047447, -162.0915680, 162.2347260
34: -97.7738495, 28.5892925, -97.8168716, 28.6727619, -126.4465942, 126.4061584
35: -91.4546967, 39.8437042, -91.5151596, 39.8598366, -131.3145294, 131.3588562
36: -90.0048676, 45.6543465, -90.0477982, 45.5900459, -135.5949097, 135.7021484
37: -131.4782715, 40.3726425, -131.4788971, 40.4203072, -171.8985596, 171.8515320
38: -106.7287521, 49.8186302, -106.7949448, 49.7047653, -156.4335175, 156.6135712
39: -118.5650024, 57.3252831, -118.6134186, 57.2662888, -175.8312683, 175.9386902
40: -100.1987305, 35.4124527, -100.2368851, 35.3569450, -135.5556793, 135.6493378
41: -84.2008972, 51.1894302, -84.2292938, 51.1639938, -135.3648987, 135.4187317
42: -66.2860718, 38.0308113, -66.3865128, 38.0029144, -104.2889786, 104.4173279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1651

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1433764, upper bound: 70.2591269
time: 91.78 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1814746, upper bound: 70.2591269
time: 194.38 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -91.4231949, 65.9274445, -91.3414612, 65.7589264, -157.1821136, 157.2689056
1: -45.6104927, 56.2468758, -45.5744400, 56.0128326, -101.6233215, 101.8213043
2: -39.9058342, 57.3188934, -39.8656464, 57.0488968, -96.9547272, 97.1845398
3: -49.9572754, 59.3869667, -49.9363861, 59.0857391, -109.0430145, 109.3233490
4: -48.8036957, 73.2746429, -48.7441978, 72.9615784, -121.7652740, 122.0188446
5: -46.2285194, 58.2747040, -46.1787910, 58.0046806, -104.2331924, 104.4534912
6: -90.9594345, 43.7874985, -90.8164444, 43.7086525, -134.6680603, 134.6039429
7: -54.9012642, 56.8437004, -54.8254395, 56.6445122, -111.5457764, 111.6691437
8: -60.7023277, 82.9222412, -60.6323891, 82.5949173, -143.2972412, 143.5546265
9: -49.6122551, 63.6138000, -49.4192314, 63.4818344, -113.0940857, 113.0330353
10: -76.9479370, 71.9319000, -76.4985809, 71.8131561, -148.7610931, 148.4304810
11: -81.0149002, 37.4584274, -80.5581894, 37.4250259, -118.4399261, 118.0165939
12: -85.1330490, 51.2358627, -84.7072906, 51.1249313, -136.2579803, 135.9431458
13: -77.4561234, 80.7627945, -77.4695511, 80.6039734, -158.0600891, 158.2323303
14: -117.5981140, 55.5610428, -117.2103195, 55.5002441, -173.0983429, 172.7713623
15: -60.4767685, 63.2925262, -60.3798904, 63.1190186, -123.5957870, 123.6724091
16: -79.3296280, 54.7541771, -79.0410461, 54.6497688, -133.9794006, 133.7952271
17: -110.8426590, 47.7350998, -110.5971985, 47.6729240, -158.5155792, 158.3322906
18: -79.0066528, 54.2645111, -78.7746658, 54.2552109, -133.2618713, 133.0391693
19: -57.8652802, 36.0203934, -57.6215324, 36.0282822, -93.8935547, 93.6419220
20: -56.5407600, 39.7201614, -56.3338699, 39.7027588, -96.2435150, 96.0540161
21: -74.2385788, 41.4692345, -73.8748932, 41.4574089, -115.6959839, 115.3441315
22: -69.0659790, 44.0143585, -68.8955002, 43.9624863, -113.0284653, 112.9098587
23: -61.6241035, 46.6240730, -61.4407539, 46.6298828, -108.2539749, 108.0648193
24: -73.3944321, 46.1026077, -73.2749710, 46.1365509, -119.5309830, 119.3775787
25: -64.1835098, 47.4485855, -64.0543823, 47.4286652, -111.6121750, 111.5029678
26: -83.0472260, 61.7238083, -82.7538605, 61.6890144, -144.7362366, 144.4776611
27: -69.3302765, 45.8881607, -69.1672211, 45.9004860, -115.2307587, 115.0553818
28: -58.3849716, 48.7676773, -58.2466393, 48.7768402, -107.1618042, 107.0143127
29: -75.1546631, 42.1833725, -74.9580841, 42.1685333, -117.3231735, 117.1414490
30: -79.1076965, 47.8001518, -78.8929291, 47.7969437, -126.9046402, 126.6930618
31: -80.2775116, 47.7499275, -80.0039597, 47.7554588, -128.0329742, 127.7538910
32: -83.5926132, 42.7152405, -83.3823242, 42.6294403, -126.2220459, 126.0975647
33: -109.7489777, 52.2012634, -109.6653900, 52.0148048, -161.7637634, 161.8666534
34: -97.7483978, 28.5277824, -97.6730652, 28.4206085, -126.1690063, 126.2008514
35: -91.4393463, 39.7403526, -91.3789825, 39.5900269, -131.0293579, 131.1193237
36: -89.9937057, 45.5576897, -89.9303436, 45.4882431, -135.4819489, 135.4880371
37: -131.4284210, 40.3973541, -131.2709351, 40.3472672, -171.7756958, 171.6682892
38: -106.7156754, 49.6674004, -106.6256638, 49.5370026, -156.2526855, 156.2930603
39: -118.5467529, 57.2303581, -118.4467545, 57.1079178, -175.6546631, 175.6771088
40: -100.1765442, 35.3236847, -100.0324936, 35.2329102, -135.4094391, 135.3561707
41: -84.1959991, 51.1484985, -84.1060944, 51.1007309, -135.2966919, 135.2545929
42: -66.2952805, 38.0616035, -66.1499252, 37.9715881, -104.2668610, 104.2115173

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1651

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1433764, upper bound: 70.2390084
time: 81.65 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1807179, upper bound: 70.2390084
time: 101.73 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -91.6335754, 66.0331268, -91.3206635, 65.7486572, -157.3822327, 157.3537903
1: -45.7495880, 56.5538025, -45.5641441, 56.0090599, -101.7586365, 102.1179352
2: -40.0525055, 57.6496658, -39.8590317, 57.0437202, -97.0962219, 97.5086975
3: -50.1038055, 59.8536415, -49.9289474, 59.0780334, -109.1818390, 109.7825928
4: -48.9953384, 73.7127380, -48.7371025, 72.9558792, -121.9512177, 122.4498367
5: -46.3878250, 58.6292267, -46.1721268, 57.9984245, -104.3862457, 104.8013535
6: -91.1364899, 43.8425369, -90.8058929, 43.6587372, -134.7952271, 134.6484375
7: -55.0938339, 57.0448380, -54.8173943, 56.6396790, -111.7335052, 111.8622284
8: -60.8558846, 83.3213806, -60.6259193, 82.5867615, -143.4426422, 143.9472961
9: -49.7312126, 63.7613602, -49.4041977, 63.4711304, -113.2023315, 113.1655579
10: -77.5687103, 72.2008209, -76.4917908, 71.8021240, -149.3708344, 148.6926117
11: -81.9343643, 37.6368980, -80.5451279, 37.4164925, -119.3508606, 118.1820221
12: -85.5648499, 51.4417191, -84.7010498, 51.1164856, -136.6813354, 136.1427612
13: -77.5787964, 81.2242279, -77.4356766, 80.5901108, -158.1689148, 158.6598816
14: -118.3874283, 55.7657471, -117.1976089, 55.4925003, -173.8799133, 172.9633484
15: -60.6357422, 63.6287346, -60.3429337, 63.1104774, -123.7462158, 123.9716644
16: -79.8817749, 54.9045334, -79.0233383, 54.6057625, -134.4875336, 133.9278564
17: -111.5668106, 47.9139328, -110.5892715, 47.6617012, -159.2285156, 158.5032043
18: -79.6167450, 54.4111671, -78.7629471, 54.2352104, -133.8519592, 133.1741028
19: -58.4245682, 36.1498413, -57.6153679, 36.0231705, -94.4477234, 93.7652130
20: -56.9203644, 39.8332176, -56.3284073, 39.6941986, -96.6145630, 96.1616211
21: -74.9756775, 41.6373177, -73.8677521, 41.4524803, -116.4281616, 115.5050659
22: -69.3722076, 44.1351166, -68.8897095, 43.9491386, -113.3213501, 113.0248108
23: -62.1204033, 46.7697449, -61.4349251, 46.6229134, -108.7433167, 108.2046661
24: -73.7369690, 46.1821747, -73.2612534, 46.1179008, -119.8548584, 119.4434280
25: -64.5658340, 47.5756226, -64.0441437, 47.4207306, -111.9865570, 111.6197510
26: -83.6041412, 61.9531059, -82.7435608, 61.6795425, -145.2836914, 144.6966705
27: -69.6053543, 45.9628525, -69.1562424, 45.8898354, -115.4951935, 115.1190948
28: -58.7043762, 48.8859520, -58.2418518, 48.7690277, -107.4733887, 107.1278000
29: -75.6213989, 42.2966919, -74.9513321, 42.1611633, -117.7825623, 117.2480240
30: -79.6667633, 47.9695663, -78.8801270, 47.7880936, -127.4548569, 126.8496933
31: -80.9738693, 47.8969955, -79.9956284, 47.7486115, -128.7224731, 127.8926239
32: -83.7213287, 42.8144379, -83.3611298, 42.6080208, -126.3293457, 126.1755676
33: -109.9264984, 52.6148453, -109.6524887, 52.0091209, -161.9356232, 162.2673340
34: -97.8812637, 28.7440681, -97.6634979, 28.4113884, -126.2926483, 126.4075623
35: -91.5343323, 40.0240440, -91.3593292, 39.5864563, -131.1207886, 131.3833618
36: -90.1105652, 45.7592316, -89.9119873, 45.4847107, -135.5952606, 135.6712189
37: -131.6324463, 40.5128365, -131.2482300, 40.3178978, -171.9503174, 171.7610474
38: -106.9142914, 49.9484482, -106.6113358, 49.5304985, -156.4447937, 156.5597839
39: -118.7412949, 57.4614334, -118.4261093, 57.1034164, -175.8446960, 175.8875427
40: -100.3780212, 35.5142822, -100.0191269, 35.2273445, -135.6053467, 135.5333862
41: -84.3398285, 51.2804070, -84.0894394, 51.0830994, -135.4229279, 135.3698425
42: -66.4504471, 38.1887283, -66.1447830, 37.9278107, -104.3782578, 104.3335037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1651

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1433764, upper bound: 70.2388843
time: 99.22 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1807179, upper bound: 70.2388843
time: 95.50 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -91.4498901, 65.9249344, -91.5659714, 65.8604584, -157.3103485, 157.4909058
1: -45.6256371, 56.2542305, -45.7152710, 56.1650085, -101.7906494, 101.9694977
2: -39.9381180, 57.3226395, -40.0307465, 57.2528305, -97.1909332, 97.3533859
3: -49.9822006, 59.3965302, -50.0792084, 59.2826157, -109.2648087, 109.4757233
4: -48.8227043, 73.2809906, -48.9251480, 73.2013016, -122.0239716, 122.2061386
5: -46.2549896, 58.2871704, -46.3390579, 58.1742630, -104.4292526, 104.6262207
6: -90.9739532, 43.7812843, -91.0763397, 43.8163910, -134.7903442, 134.8576202
7: -54.9164124, 56.8515549, -55.0216866, 56.7495499, -111.6659622, 111.8732452
8: -60.7433968, 82.9340820, -60.8390465, 82.8671265, -143.6105194, 143.7731018
9: -49.6203651, 63.6642570, -49.7067223, 63.6885147, -113.3088684, 113.3709793
10: -76.9662018, 72.0352020, -77.1728668, 72.1725693, -149.1387634, 149.2080688
11: -81.0368195, 37.5273285, -81.2259674, 37.6655235, -118.7023392, 118.7532806
12: -85.1435394, 51.3278122, -85.2255859, 51.4658127, -136.6093445, 136.5533905
13: -77.4606323, 80.8024597, -77.5905838, 80.8726959, -158.3333282, 158.3930359
14: -117.6100693, 55.6665268, -117.8430481, 55.8379364, -173.4479980, 173.5095673
15: -60.5323448, 63.3080864, -60.6520996, 63.4109573, -123.9432983, 123.9601822
16: -79.3456268, 54.8192291, -79.5523453, 54.9019699, -134.2475891, 134.3715820
17: -110.8498383, 47.7668190, -110.9131088, 47.9052353, -158.7550659, 158.6799316
18: -79.0137329, 54.2859917, -79.0505524, 54.4026260, -133.4163208, 133.3365479
19: -57.8766479, 36.0363770, -57.8119164, 36.1339340, -94.0105820, 93.8482971
20: -56.5586052, 39.7599030, -56.6112099, 39.8495407, -96.4081421, 96.3711014
21: -74.2517395, 41.5134544, -74.2783661, 41.6399574, -115.8916931, 115.7918167
22: -69.0836334, 44.0282097, -69.1129913, 44.1726761, -113.2563019, 113.1411972
23: -61.6345100, 46.6435661, -61.6069984, 46.7431908, -108.3777008, 108.2505646
24: -73.4339294, 46.1014977, -73.4858322, 46.1850739, -119.6189880, 119.5873184
25: -64.1915894, 47.4722366, -64.1888123, 47.5759239, -111.7675171, 111.6610489
26: -83.0652924, 61.7771912, -83.0334778, 61.9348030, -145.0000916, 144.8106689
27: -69.3865051, 45.8932266, -69.4455719, 45.9834747, -115.3699722, 115.3387909
28: -58.3994980, 48.7712479, -58.3845291, 48.8486137, -107.2481079, 107.1557770
29: -75.1722565, 42.1938553, -75.1621246, 42.2965012, -117.4687500, 117.3559799
30: -79.1250458, 47.8179626, -79.0614166, 47.9358177, -127.0608521, 126.8793793
31: -80.2976990, 47.7900085, -80.2830582, 47.9167328, -128.2144165, 128.0730591
32: -83.6093140, 42.7512054, -83.6927795, 42.7797623, -126.3890686, 126.4439697
33: -109.8089142, 52.2154045, -109.9226913, 52.3357925, -162.1446991, 162.1380920
34: -97.7888184, 28.5377407, -97.8677902, 28.7043648, -126.4931793, 126.4055328
35: -91.4847183, 39.7537842, -91.5738525, 39.8835983, -131.3683167, 131.3276367
36: -90.0101013, 45.5652351, -90.0877380, 45.6112099, -135.6213074, 135.6529694
37: -131.4619751, 40.4102592, -131.5296936, 40.4868584, -171.9488068, 171.9399414
38: -106.7304001, 49.6862259, -106.8499527, 49.7317619, -156.4621582, 156.5361786
39: -118.5653458, 57.2652550, -118.6740952, 57.3063316, -175.8716736, 175.9393311
40: -100.2029419, 35.3288612, -100.2829056, 35.3706284, -135.5735779, 135.6117554
41: -84.2087402, 51.1535378, -84.2736816, 51.1939621, -135.4027100, 135.4272156
42: -66.3095169, 38.0427895, -66.4123535, 38.0636673, -104.3731842, 104.4551392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1651

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1442196, upper bound: 70.2593461
time: 111.05 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1807179, upper bound: 70.2593461
time: 85.00 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -91.6609421, 66.0305481, -91.5451431, 65.8502350, -157.5111694, 157.5756836
1: -45.7652245, 56.5611801, -45.7049141, 56.1612396, -101.9264374, 102.2660904
2: -40.0858383, 57.6533585, -40.0241203, 57.2476234, -97.3334656, 97.6774750
3: -50.1291199, 59.8633270, -50.0718079, 59.2749214, -109.4040375, 109.9351349
4: -49.0145454, 73.7191010, -48.9179955, 73.1956482, -122.2101898, 122.6371002
5: -46.4147911, 58.6417618, -46.3324165, 58.1679955, -104.5827866, 104.9741821
6: -91.1512756, 43.8362312, -91.0657959, 43.7663994, -134.9176788, 134.9020233
7: -55.1097832, 57.0528069, -55.0135498, 56.7446976, -111.8544769, 112.0663528
8: -60.8975410, 83.3331909, -60.8325500, 82.8590546, -143.7565918, 144.1657410
9: -49.7395706, 63.8123283, -49.6917648, 63.6776428, -113.4172134, 113.5040894
10: -77.5870361, 72.3050537, -77.1660767, 72.1614914, -149.7485352, 149.4711304
11: -81.9562378, 37.7063713, -81.2128830, 37.6570091, -119.6132355, 118.9192505
12: -85.5752945, 51.5340538, -85.2193451, 51.4572601, -137.0325317, 136.7534027
13: -77.5835114, 81.2632980, -77.5565872, 80.8586273, -158.4421387, 158.8198853
14: -118.3995132, 55.8713341, -117.8302307, 55.8300934, -174.2295990, 173.7015686
15: -60.6919403, 63.6443596, -60.6151428, 63.4027748, -124.0947113, 124.2595062
16: -79.8977280, 54.9703674, -79.5349426, 54.8579407, -134.7556458, 134.5053101
17: -111.5741196, 47.9458809, -110.9051132, 47.8939018, -159.4680176, 158.8509979
18: -79.6237640, 54.4329643, -79.0386734, 54.3825264, -134.0062866, 133.4716339
19: -58.4366722, 36.1662025, -57.8057518, 36.1288376, -94.5654907, 93.9719467
20: -56.9382057, 39.8733597, -56.6057587, 39.8409462, -96.7791519, 96.4791183
21: -74.9888916, 41.6822205, -74.2713013, 41.6350060, -116.6238861, 115.9535141
22: -69.3898849, 44.1496048, -69.1072006, 44.1593170, -113.5491943, 113.2568054
23: -62.1308250, 46.7897644, -61.6011581, 46.7362289, -108.8670502, 108.3909149
24: -73.7763519, 46.1810951, -73.4720306, 46.1664238, -119.9427795, 119.6531219
25: -64.5735016, 47.5997162, -64.1784897, 47.5679169, -112.1414185, 111.7782059
26: -83.6222763, 62.0068550, -83.0231094, 61.9252892, -145.5475616, 145.0299683
27: -69.6613617, 45.9680138, -69.4344940, 45.9728699, -115.6342316, 115.4025040
28: -58.7189407, 48.8896942, -58.3796844, 48.8407822, -107.5597000, 107.2693787
29: -75.6388702, 42.3076019, -75.1553802, 42.2892609, -117.9281311, 117.4629822
30: -79.6840744, 47.9874115, -79.0485611, 47.9268799, -127.6109467, 127.0359726
31: -80.9940567, 47.9376297, -80.2748108, 47.9098740, -128.9039307, 128.2124329
32: -83.7381668, 42.8509102, -83.6716766, 42.7580109, -126.4961777, 126.5225830
33: -109.9864960, 52.6290436, -109.9098053, 52.3300476, -162.3165283, 162.5388489
34: -97.9217224, 28.7542973, -97.8580627, 28.6951885, -126.6169128, 126.6123581
35: -91.5798264, 40.0376549, -91.5541687, 39.8801422, -131.4599609, 131.5918121
36: -90.1271057, 45.7669296, -90.0692902, 45.6075745, -135.7346802, 135.8362122
37: -131.6656189, 40.5258636, -131.5070190, 40.4575462, -172.1231689, 172.0328674
38: -106.9294434, 49.9671936, -106.8355942, 49.7252541, -156.6546936, 156.8027954
39: -118.7601242, 57.4964142, -118.6533051, 57.3018036, -176.0619202, 176.1497192
40: -100.4049606, 35.5193634, -100.2695770, 35.3650169, -135.7699738, 135.7889252
41: -84.3533020, 51.2854729, -84.2565308, 51.1763725, -135.5296783, 135.5419922
42: -66.4648743, 38.1680565, -66.4072113, 38.0198097, -104.4846802, 104.5752716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=456, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=544, inp2_unstable=544, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1651

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1433764, upper bound: 70.2591269
time: 89.48 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1807179, upper bound: 70.2591269
time: 97.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 189.19 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1433764, upper bound: 70.2390084
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1807179, upper bound: 70.2390084
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1433764, upper bound: 70.2388843
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1807179, upper bound: 70.2388843
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1442196, upper bound: 70.2593461
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1814746, upper bound: 70.2593461
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1433764, upper bound: 70.2591269
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1814746, upper bound: 70.2591269
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1433764, upper bound: 70.2390084
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1807179, upper bound: 70.2390084
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1433764, upper bound: 70.2388843
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1807179, upper bound: 70.2388843
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1442196, upper bound: 70.2593461
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1807179, upper bound: 70.2593461
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1433764, upper bound: 70.2591269
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 189.19
Output dim: 4, lower bound: -70.1807179, upper bound: 70.2591269

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -91.1358185, 65.6194534, -91.0223389, 65.5365524, -156.6723633, 156.6417847
1: -45.4284592, 55.9230270, -45.3838768, 55.8749924, -101.3034515, 101.3069000
2: -39.6606598, 56.9305649, -39.6403503, 56.8694992, -96.5301590, 96.5709152
3: -49.7069817, 58.9328690, -49.6679077, 58.8502884, -108.5572586, 108.6007690
4: -48.5219879, 72.8003845, -48.4670334, 72.6970062, -121.2189789, 121.2674026
5: -45.9372253, 57.8562164, -45.9039841, 57.7726936, -103.7099152, 103.7602005
6: -90.5687408, 43.6770096, -90.4212646, 43.5116692, -134.0803986, 134.0982666
7: -54.6375427, 56.5669327, -54.6367645, 56.5138512, -111.1513824, 111.2036972
8: -60.4416466, 82.4541016, -60.3799324, 82.3751373, -142.8167877, 142.8340302
9: -49.3403244, 63.2683563, -49.1832123, 63.1663933, -112.5067139, 112.4515686
10: -76.3939819, 71.4086151, -76.2064819, 71.3808670, -147.7748413, 147.6150818
11: -80.4057236, 37.1827011, -80.3015823, 37.1537437, -117.5594635, 117.4842834
12: -84.5522156, 50.8371506, -84.4645615, 50.8111267, -135.3633423, 135.3017120
13: -77.3125610, 80.4648743, -77.2696381, 80.3527451, -157.6653137, 157.7345123
14: -117.0661011, 55.1183929, -116.9233170, 55.1323013, -172.1983948, 172.0417175
15: -60.2503777, 62.8966331, -60.0754013, 62.7818909, -123.0322723, 122.9720306
16: -78.9137955, 54.4528198, -78.8235703, 54.4393349, -133.3531189, 133.2763977
17: -110.4846344, 47.4890747, -110.3585358, 47.4902420, -157.9748840, 157.8476105
18: -78.6817245, 54.0622787, -78.6311340, 54.0899315, -132.7716522, 132.6934052
19: -57.4770088, 35.8387680, -57.4082069, 35.8335876, -93.3105927, 93.2469788
20: -56.1864510, 39.5320396, -56.1167564, 39.4833260, -95.6697693, 95.6487885
21: -73.7411957, 41.2051468, -73.6603241, 41.2039871, -114.9451752, 114.8654709
22: -68.8444519, 43.8543472, -68.7479172, 43.8529129, -112.6973648, 112.6022644
23: -61.3020134, 46.4395142, -61.2298126, 46.4256744, -107.7276917, 107.6693192
24: -73.1331406, 46.0351105, -73.0017929, 45.9703789, -119.1035156, 119.0369034
25: -63.9068260, 47.2518158, -63.8085098, 47.1985550, -111.1053772, 111.0603180
26: -82.6458282, 61.3766861, -82.5843201, 61.4407349, -144.0865479, 143.9609985
27: -69.0372925, 45.8153839, -68.9223328, 45.7814674, -114.8187561, 114.7377167
28: -58.0988731, 48.6017838, -57.9974022, 48.5700226, -106.6688995, 106.5991821
29: -74.8849106, 42.0383263, -74.7856140, 42.0515747, -116.9364700, 116.8239441
30: -78.6999130, 47.5757675, -78.5766602, 47.5159645, -126.2158737, 126.1524277
31: -79.8041687, 47.5308647, -79.7150574, 47.5281563, -127.3323135, 127.2459183
32: -83.2077332, 42.5366898, -83.1086578, 42.4374390, -125.6451645, 125.6453476
33: -109.4251709, 51.9422073, -109.3570862, 51.7876549, -161.2128296, 161.2992859
34: -97.4768982, 28.3475952, -97.3952789, 28.2272110, -125.7041016, 125.7428741
35: -91.2415619, 39.5351257, -91.1867676, 39.4203949, -130.6619568, 130.7218933
36: -89.7648163, 45.4324722, -89.7035675, 45.3465042, -135.1113281, 135.1360474
37: -131.0542603, 40.2254868, -130.8887329, 40.0885201, -171.1427765, 171.1142120
38: -106.3848953, 49.5000305, -106.3285446, 49.3482704, -155.7331696, 155.8285828
39: -118.2368240, 57.0351791, -118.1699905, 56.9176788, -175.1544952, 175.2051697
40: -99.8235092, 35.1976395, -99.7099762, 35.0722122, -134.8957062, 134.9076233
41: -83.9071960, 51.0305023, -83.8215790, 50.9267235, -134.8339233, 134.8520813
42: -65.9682007, 37.9145622, -65.8538589, 37.7981911, -103.7663879, 103.7684174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2373597
time: 92.28 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2375630
time: 88.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -91.1822205, 65.7170792, -91.2618256, 65.7219849, -156.9042053, 156.9789124
1: -45.4457664, 55.9804153, -45.5185852, 55.9900360, -101.4358063, 101.4990005
2: -39.6798897, 57.0176697, -39.7795410, 57.0274963, -96.7073822, 96.7972031
3: -49.7263947, 59.0454597, -49.8443832, 59.0584755, -108.7848663, 108.8898392
4: -48.5400925, 72.9297333, -48.6478920, 72.9324341, -121.4725266, 121.5776215
5: -45.9613724, 57.9675789, -46.0727806, 57.9758606, -103.9372177, 104.0403595
6: -90.7561874, 43.6946487, -90.7671967, 43.7024803, -134.4586639, 134.4618530
7: -54.6597710, 56.6220245, -54.7506638, 56.6229553, -111.2827148, 111.3726883
8: -60.4610596, 82.5532379, -60.5420609, 82.5648499, -143.0259094, 143.0953064
9: -49.3619537, 63.3898392, -49.3924599, 63.3985748, -112.7605286, 112.7822876
10: -76.4265747, 71.5495300, -76.4589767, 71.6550980, -148.0816650, 148.0084991
11: -80.5005722, 37.2052803, -80.5163040, 37.3230858, -117.8236542, 117.7215805
12: -84.6615753, 50.8670654, -84.6778183, 50.9754868, -135.6370544, 135.5448914
13: -77.3481445, 80.5510788, -77.4459229, 80.5451660, -157.8933105, 157.9970093
14: -117.1209717, 55.2382088, -117.1643982, 55.3593063, -172.4802704, 172.4026031
15: -60.2795715, 63.0582581, -60.3320694, 63.0723572, -123.3519287, 123.3903275
16: -78.9494095, 54.5016174, -78.9953690, 54.5603714, -133.5097656, 133.4969788
17: -110.5361786, 47.5303726, -110.5627975, 47.6107330, -158.1469116, 158.0931702
18: -78.7159119, 54.0930252, -78.7382278, 54.1945610, -132.9104767, 132.8312531
19: -57.5629768, 35.8561707, -57.5837631, 35.9634666, -93.5264282, 93.4399338
20: -56.2733116, 39.5493927, -56.2951469, 39.6328201, -95.9061127, 95.8445358
21: -73.8136444, 41.2361908, -73.8303604, 41.3660736, -115.1797104, 115.0665512
22: -68.8907700, 43.8719406, -68.8661041, 43.9302063, -112.8209763, 112.7380447
23: -61.3901176, 46.4646683, -61.4062691, 46.5713196, -107.9614410, 107.8709412
24: -73.2560883, 46.0521469, -73.2354660, 46.1252365, -119.3813248, 119.2876129
25: -64.0152206, 47.2741470, -64.0200958, 47.3680305, -111.3832550, 111.2942429
26: -82.6965485, 61.4053116, -82.7179718, 61.5647125, -144.2612610, 144.1232910
27: -69.1311798, 45.8316269, -69.1114349, 45.8864098, -115.0175781, 114.9430618
28: -58.2122231, 48.6183853, -58.2098579, 48.7193527, -106.9315643, 106.8282471
29: -74.9395523, 42.0500565, -74.9263916, 42.1276550, -117.0672073, 116.9764481
30: -78.8426056, 47.6023216, -78.8529053, 47.7270889, -126.5696869, 126.4552307
31: -79.9253387, 47.5555077, -79.9537659, 47.6773643, -127.6027069, 127.5092697
32: -83.3311615, 42.5555267, -83.3448792, 42.5711212, -125.9022827, 125.9003906
33: -109.5413055, 51.9621277, -109.5855255, 51.9843483, -161.5256500, 161.5476532
34: -97.5926208, 28.3648453, -97.6160736, 28.3930588, -125.9856796, 125.9809189
35: -91.3136826, 39.5493469, -91.3304291, 39.5652008, -130.8788757, 130.8797760
36: -89.8656845, 45.4466743, -89.8945465, 45.4687729, -135.3344574, 135.3412170
37: -131.2308960, 40.2478180, -131.2189941, 40.3079529, -171.5388489, 171.4668121
38: -106.5020065, 49.5185013, -106.5636063, 49.5132408, -156.0152435, 156.0821075
39: -118.3467102, 57.0562057, -118.3932571, 57.0698967, -175.4165955, 175.4494629
40: -99.9599304, 35.2143097, -99.9754791, 35.2221069, -135.1820374, 135.1897888
41: -84.0354004, 51.0505295, -84.0631866, 51.0822067, -135.1175995, 135.1137085
42: -66.1079254, 37.9344330, -66.1125565, 37.9610710, -104.0690002, 104.0469818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2373597
time: 894.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2375630
time: 77.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -91.3413696, 65.7232208, -91.0002213, 65.5261536, -156.8675232, 156.7234497
1: -45.5672035, 56.2282562, -45.3730659, 55.8710518, -101.4382553, 101.6013184
2: -39.8075867, 57.2596970, -39.6333275, 56.8640213, -96.6715927, 96.8930206
3: -49.8537140, 59.3971825, -49.6600914, 58.8421516, -108.6958618, 109.0572739
4: -48.7142525, 73.2366638, -48.4594803, 72.6911926, -121.4054413, 121.6961441
5: -46.0982208, 58.2074852, -45.8969803, 57.7660294, -103.8642502, 104.1044617
6: -90.7433319, 43.7267838, -90.4105225, 43.4595413, -134.2028809, 134.1372986
7: -54.8289719, 56.7609558, -54.6281548, 56.5086632, -111.3376312, 111.3891144
8: -60.5946693, 82.8516922, -60.3729858, 82.3667908, -142.9614410, 143.2246704
9: -49.4592171, 63.4167595, -49.1682243, 63.1553230, -112.6145325, 112.5849762
10: -77.0138702, 71.6788559, -76.1992493, 71.3693848, -148.3832550, 147.8781128
11: -81.3202667, 37.3623390, -80.2879410, 37.1448708, -118.4651337, 117.6502838
12: -84.9703598, 51.0424614, -84.4581070, 50.8019180, -135.7722473, 135.5005646
13: -77.4163513, 80.9185638, -77.2249756, 80.3381577, -157.7545166, 158.1435242
14: -117.8415451, 55.3209152, -116.9100571, 55.1240120, -172.9655609, 172.2309723
15: -60.3727608, 63.2288666, -60.0199852, 62.7732506, -123.1459885, 123.2488556
16: -79.4393616, 54.6009865, -78.8052368, 54.3949852, -133.8343506, 133.4062195
17: -111.1977310, 47.6669693, -110.3501511, 47.4786606, -158.6763916, 158.0171051
18: -79.2827377, 54.2090225, -78.6189575, 54.0695763, -133.3522644, 132.8279724
19: -58.0340881, 35.9701653, -57.4017143, 35.8282089, -93.8622971, 93.3718719
20: -56.5638695, 39.6437912, -56.1110687, 39.4740944, -96.0379639, 95.7548523
21: -74.4757080, 41.3737717, -73.6529083, 41.1985359, -115.6742401, 115.0266800
22: -69.1489182, 43.9691505, -68.7417755, 43.8389435, -112.9878616, 112.7109222
23: -61.7941284, 46.5862808, -61.2236252, 46.4184685, -108.2126007, 107.8099060
24: -73.4710541, 46.1029434, -72.9872589, 45.9464645, -119.4175034, 119.0902023
25: -64.2851715, 47.3765488, -63.7976112, 47.1902504, -111.4754105, 111.1741486
26: -83.1858521, 61.6071625, -82.5733795, 61.4307022, -144.6165314, 144.1805420
27: -69.3148270, 45.8864441, -68.9109192, 45.7687263, -115.0835571, 114.7973633
28: -58.4149895, 48.7215385, -57.9923553, 48.5619202, -106.9769135, 106.7138977
29: -75.3432083, 42.1461792, -74.7785721, 42.0429688, -117.3861771, 116.9247360
30: -79.2546234, 47.7456284, -78.5633087, 47.5068512, -126.7614746, 126.3089218
31: -80.4945221, 47.6787338, -79.7064667, 47.5209198, -128.0154419, 127.3851852
32: -83.3361511, 42.6311569, -83.0873413, 42.4149475, -125.7510986, 125.7184982
33: -109.6006317, 52.3498611, -109.3436737, 51.7816963, -161.3823242, 161.6935272
34: -97.6074829, 28.5575409, -97.3851929, 28.2177696, -125.8252411, 125.9427338
35: -91.3295746, 39.8125153, -91.1644592, 39.4165726, -130.7461395, 130.9769745
36: -89.8769531, 45.6312027, -89.6829224, 45.3424835, -135.2194366, 135.3141174
37: -131.2530518, 40.3336258, -130.8648987, 40.0548553, -171.3079071, 171.1985168
38: -106.5846329, 49.7783852, -106.3138351, 49.3414421, -155.9260406, 156.0922241
39: -118.4277039, 57.2671089, -118.1485062, 56.9129753, -175.3406677, 175.4156036
40: -100.0194397, 35.3877831, -99.6959686, 35.0664635, -135.0858917, 135.0837555
41: -84.0492706, 51.1600685, -83.8042374, 50.9087830, -134.9580383, 134.9643097
42: -66.1212845, 38.0315094, -65.8484650, 37.7529449, -103.8742294, 103.8799744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2372314
time: 73.30 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2374416
time: 86.47 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -91.3879013, 65.8208694, -91.2396927, 65.7115784, -157.0994873, 157.0605621
1: -45.5845718, 56.2856483, -45.5077858, 55.9860916, -101.5706635, 101.7934265
2: -39.8268967, 57.3467102, -39.7725372, 57.0220032, -96.8488998, 97.1192474
3: -49.8731842, 59.5097580, -49.8366051, 59.0503387, -108.9235229, 109.3463593
4: -48.7324028, 73.3660202, -48.6403847, 72.9266357, -121.6590271, 122.0064087
5: -46.1224823, 58.3187866, -46.0657578, 57.9692154, -104.0916901, 104.3845367
6: -90.9306488, 43.7443771, -90.7564545, 43.6503372, -134.5809937, 134.5008240
7: -54.8513069, 56.8160248, -54.7421112, 56.6178207, -111.4691315, 111.5581360
8: -60.6141739, 82.9507751, -60.5351257, 82.5564117, -143.1705933, 143.4858856
9: -49.4808388, 63.5382004, -49.3773956, 63.3875046, -112.8683472, 112.9155884
10: -77.0463715, 71.8196640, -76.4517365, 71.6435699, -148.6899261, 148.2713928
11: -81.4150772, 37.3849411, -80.5026321, 37.3142052, -118.7292786, 117.8875732
12: -85.0797272, 51.0725327, -84.6713867, 50.9662743, -136.0459900, 135.7439117
13: -77.4519730, 81.0046539, -77.4012222, 80.5305862, -157.9825439, 158.4058685
14: -117.8963013, 55.4406967, -117.1511841, 55.3510132, -173.2473145, 172.5918884
15: -60.4020538, 63.3904953, -60.2766953, 63.0636978, -123.4657516, 123.6671906
16: -79.4748764, 54.6497879, -78.9770126, 54.5160065, -133.9908752, 133.6268005
17: -111.2489853, 47.7081985, -110.5544357, 47.5990829, -158.8480377, 158.2626343
18: -79.3168564, 54.2397881, -78.7260361, 54.1741791, -133.4910278, 132.9658203
19: -58.1199646, 35.9876175, -57.5772514, 35.9580612, -94.0780258, 93.5648651
20: -56.6507111, 39.6611786, -56.2894859, 39.6236191, -96.2743301, 95.9506531
21: -74.5480728, 41.4049110, -73.8229523, 41.3606339, -115.9087067, 115.2278519
22: -69.1950531, 43.9867821, -68.8600159, 43.9162445, -113.1112976, 112.8467941
23: -61.8821411, 46.6114807, -61.4000473, 46.5641327, -108.4462738, 108.0115280
24: -73.5939484, 46.1200180, -73.2209549, 46.1012726, -119.6951828, 119.3409729
25: -64.3935776, 47.3989296, -64.0092316, 47.3597260, -111.7533035, 111.4081421
26: -83.2364883, 61.6358757, -82.7070007, 61.5547256, -144.7912140, 144.3428802
27: -69.4085541, 45.9027557, -69.1000290, 45.8736801, -115.2822266, 115.0027847
28: -58.5282402, 48.7381363, -58.2048454, 48.7112808, -107.2395172, 106.9429779
29: -75.3977356, 42.1579361, -74.9193268, 42.1190796, -117.5168076, 117.0772629
30: -79.3972931, 47.7722664, -78.8395767, 47.7179565, -127.1152420, 126.6118393
31: -80.6156311, 47.7033653, -79.9451141, 47.6701355, -128.2857513, 127.6484833
32: -83.4595490, 42.6500626, -83.3235168, 42.5485992, -126.0081406, 125.9735489
33: -109.7167511, 52.3697624, -109.5720978, 51.9783592, -161.6950989, 161.9418640
34: -97.7231827, 28.5748787, -97.6060181, 28.3836117, -126.1067963, 126.1808929
35: -91.4017029, 39.8267555, -91.3081589, 39.5614014, -130.9631042, 131.1349182
36: -89.9777374, 45.6454544, -89.8738480, 45.4647675, -135.4425049, 135.5193024
37: -131.4295349, 40.3559952, -131.1950378, 40.2743378, -171.7038574, 171.5510254
38: -106.7017670, 49.7968864, -106.5489044, 49.5064621, -156.2082214, 156.3457947
39: -118.5375671, 57.2880783, -118.3717194, 57.0651779, -175.6027527, 175.6597900
40: -100.1556168, 35.4044609, -99.9614868, 35.2163658, -135.3719788, 135.3659515
41: -84.1773987, 51.1801147, -84.0459290, 51.0642776, -135.2416687, 135.2260437
42: -66.2608795, 38.0514336, -66.1071472, 37.9158096, -104.1766815, 104.1585770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2372314
time: 104.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2374416
time: 92.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -91.1626434, 65.6184769, -91.2468796, 65.6377716, -156.8004150, 156.8653564
1: -45.4440651, 55.9302902, -45.5252762, 56.0271072, -101.4711685, 101.4555664
2: -39.6904449, 56.9344940, -39.8038864, 57.0736771, -96.7641220, 96.7383804
3: -49.7276344, 58.9425354, -49.8023834, 59.0472946, -108.7749176, 108.7449188
4: -48.5422478, 72.8069534, -48.6483002, 72.9364166, -121.4786682, 121.4552536
5: -45.9578705, 57.8690453, -46.0595589, 57.9425049, -103.9003754, 103.9286041
6: -90.5833282, 43.6720009, -90.6809006, 43.6178436, -134.2011719, 134.3529053
7: -54.6505966, 56.5754852, -54.8336143, 56.6186638, -111.2692490, 111.4091034
8: -60.4833755, 82.4660797, -60.5868225, 82.6478577, -143.1312256, 143.0529022
9: -49.3488274, 63.3198013, -49.4706039, 63.3735275, -112.7223511, 112.7904053
10: -76.4126282, 71.5133591, -76.8811493, 71.7418289, -148.1544495, 148.3945007
11: -80.4278946, 37.2520905, -80.9699936, 37.3941040, -117.8219986, 118.2220688
12: -84.5628815, 50.9298439, -84.9827499, 51.1523209, -135.7151947, 135.9125977
13: -77.3173752, 80.5056992, -77.3908615, 80.6201019, -157.9374695, 157.8965607
14: -117.0800095, 55.2239609, -117.5560837, 55.4695892, -172.5495911, 172.7800446
15: -60.3061218, 62.9123878, -60.3473015, 63.0745697, -123.3806915, 123.2596893
16: -78.9304657, 54.5191765, -79.3353882, 54.6917610, -133.6222229, 133.8545685
17: -110.4926834, 47.5203934, -110.6749115, 47.7210274, -158.2137146, 158.1952972
18: -78.6904984, 54.0819054, -78.9067841, 54.2349663, -132.9254608, 132.9886932
19: -57.4893761, 35.8558350, -57.5986786, 35.9394379, -93.4287949, 93.4545135
20: -56.2042885, 39.5723419, -56.3941574, 39.6305618, -95.8348465, 95.9664993
21: -73.7546844, 41.2502289, -74.0639496, 41.3868790, -115.1415558, 115.3141632
22: -68.8617859, 43.8683395, -68.9647675, 44.0627975, -112.9245834, 112.8331070
23: -61.3127899, 46.4597321, -61.3958969, 46.5392838, -107.8520737, 107.8556290
24: -73.1731339, 46.0327911, -73.2122498, 46.0173492, -119.1904831, 119.2450333
25: -63.9181633, 47.2756042, -63.9426384, 47.3455925, -111.2637558, 111.2182465
26: -82.6641693, 61.4235153, -82.8632202, 61.6791611, -144.3433228, 144.2867432
27: -69.0942764, 45.8205070, -69.2011414, 45.8646126, -114.9588928, 115.0216522
28: -58.1143379, 48.6048965, -58.1351013, 48.6414566, -106.7557983, 106.7399979
29: -74.9062653, 42.0488777, -74.9891815, 42.1806259, -117.0868912, 117.0380402
30: -78.7177277, 47.5945587, -78.7451935, 47.6543732, -126.3721008, 126.3397446
31: -79.8247986, 47.5714264, -79.9946289, 47.6892090, -127.5139999, 127.5660477
32: -83.2247238, 42.5738831, -83.4192200, 42.5882378, -125.8129578, 125.9931030
33: -109.4852753, 51.9580040, -109.6144791, 52.1087112, -161.5939941, 161.5724792
34: -97.5176544, 28.3586502, -97.5901566, 28.5111580, -126.0288086, 125.9488068
35: -91.2870789, 39.5495377, -91.3817902, 39.7145882, -131.0016632, 130.9313354
36: -89.7811890, 45.4384232, -89.8604736, 45.4674683, -135.2486572, 135.2988892
37: -131.0880432, 40.2386932, -131.1468201, 40.2292938, -171.3173370, 171.3854980
38: -106.4002762, 49.5191269, -106.5562439, 49.5420265, -155.9423065, 156.0753784
39: -118.2532196, 57.0690498, -118.3947906, 57.1140976, -175.3673096, 175.4638367
40: -99.8498611, 35.2028618, -99.9605789, 35.2084808, -135.0583496, 135.1634369
41: -83.9200516, 51.0370636, -83.9889984, 51.0214767, -134.9415283, 135.0260620
42: -65.9825592, 37.8909607, -66.1158447, 37.8801003, -103.8626556, 104.0068054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1036525, upper bound: 70.2577376
time: 80.55 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1036525, upper bound: 70.2579013
time: 77.43 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -91.2090759, 65.7160873, -91.4865189, 65.8231964, -157.0322723, 157.2026062
1: -45.4613953, 55.9876404, -45.6600380, 56.1421165, -101.6035156, 101.6476746
2: -39.7096748, 57.0215836, -39.9431839, 57.2316055, -96.9412842, 96.9647675
3: -49.7471161, 59.0551186, -49.9789886, 59.2554474, -109.0025635, 109.0341034
4: -48.5604172, 72.9362717, -48.8292885, 73.1718140, -121.7322311, 121.7655640
5: -45.9820023, 57.9803581, -46.2284470, 58.1456299, -104.1276321, 104.2088013
6: -90.7707825, 43.6896019, -91.0266571, 43.8087082, -134.5794983, 134.7162476
7: -54.6728172, 56.6305923, -54.9476128, 56.7277794, -111.4005966, 111.5782013
8: -60.5028267, 82.5652924, -60.7490311, 82.8374176, -143.3402405, 143.3143005
9: -49.3703957, 63.4412613, -49.6798019, 63.6057129, -112.9761047, 113.1210632
10: -76.4451981, 71.6542130, -77.1335907, 72.0160141, -148.4612122, 148.7877960
11: -80.5227509, 37.2746887, -81.1843491, 37.5634727, -118.0862274, 118.4590149
12: -84.6722412, 50.9597778, -85.1959381, 51.3168144, -135.9890594, 136.1557007
13: -77.3529358, 80.5918732, -77.5671692, 80.8123779, -158.1653137, 158.1590424
14: -117.1349030, 55.3437805, -117.7970734, 55.6965218, -172.8314209, 173.1408539
15: -60.3353043, 63.0740013, -60.6041107, 63.3649673, -123.7002716, 123.6781158
16: -78.9660873, 54.5680161, -79.5071106, 54.8127365, -133.7788086, 134.0751343
17: -110.5442429, 47.5617180, -110.8789597, 47.8417053, -158.3859558, 158.4406738
18: -78.7247162, 54.1126785, -79.0139465, 54.3397141, -133.0643921, 133.1266174
19: -57.5753174, 35.8732262, -57.7741814, 36.0694427, -93.6447525, 93.6473999
20: -56.2911453, 39.5896683, -56.5724754, 39.7801132, -96.0712509, 96.1621323
21: -73.8271027, 41.2812958, -74.2338104, 41.5491943, -115.3762817, 115.5151062
22: -68.9080582, 43.8859711, -69.0829849, 44.1401863, -113.0482407, 112.9689560
23: -61.4008789, 46.4849091, -61.5722771, 46.6850395, -108.0859222, 108.0571899
24: -73.2960815, 46.0498352, -73.4459686, 46.1722565, -119.4683228, 119.4957962
25: -64.0266113, 47.2979279, -64.1543427, 47.5152206, -111.5418167, 111.4522629
26: -82.7148819, 61.4521484, -82.9967957, 61.8032875, -144.5181732, 144.4489441
27: -69.1881714, 45.8367844, -69.3901215, 45.9696121, -115.1577835, 115.2269058
28: -58.2277107, 48.6214981, -58.3476105, 48.7908173, -107.0185242, 106.9691086
29: -74.9609222, 42.0606308, -75.1298981, 42.2567139, -117.2176361, 117.1905136
30: -78.8603821, 47.6211891, -79.0214005, 47.8656044, -126.7259750, 126.6425934
31: -79.9459686, 47.5960274, -80.2332153, 47.8384781, -127.7844467, 127.8292389
32: -83.3481750, 42.5927238, -83.6552734, 42.7219391, -126.0701141, 126.2480011
33: -109.6013641, 51.9779396, -109.8429337, 52.3054733, -161.9068298, 161.8208771
34: -97.6333694, 28.3759651, -97.8108978, 28.6770515, -126.3104248, 126.1868591
35: -91.3592300, 39.5637398, -91.5254669, 39.8594475, -131.2186737, 131.0892029
36: -89.8820724, 45.4526443, -90.0514069, 45.5898285, -135.4718933, 135.5040436
37: -131.2646179, 40.2610474, -131.4771118, 40.4488068, -171.7134247, 171.7381439
38: -106.5174103, 49.5376625, -106.7913284, 49.7071304, -156.2245483, 156.3289795
39: -118.3631134, 57.0900726, -118.6180573, 57.2663918, -175.6295013, 175.7081146
40: -99.9862976, 35.2195587, -100.2257233, 35.3584747, -135.3447723, 135.4452820
41: -84.0482483, 51.0570755, -84.2305298, 51.1770287, -135.2252502, 135.2875977
42: -66.1222534, 37.9108162, -66.3743744, 38.0430298, -104.1652679, 104.2851868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1409378, upper bound: 70.2577376
time: 89.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1409378, upper bound: 70.2579013
time: 77.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -91.3678894, 65.7221375, -91.2246704, 65.6274261, -156.9953003, 156.9468079
1: -45.5831604, 56.2354660, -45.5144043, 56.0231857, -101.6063461, 101.7498703
2: -39.8378487, 57.2635345, -39.7968864, 57.0681915, -96.9060364, 97.0604248
3: -49.8747597, 59.4068947, -49.7946053, 59.0391502, -108.9138947, 109.2014999
4: -48.7345238, 73.2432251, -48.6407280, 72.9306488, -121.6651764, 121.8839569
5: -46.1193848, 58.2203445, -46.0525589, 57.9358406, -104.0552216, 104.2729034
6: -90.7581482, 43.7215881, -90.6701965, 43.5656891, -134.3238373, 134.3917847
7: -54.8430901, 56.7696686, -54.8249512, 56.6134949, -111.4565887, 111.5946198
8: -60.6368790, 82.8636322, -60.5798645, 82.6394806, -143.2763367, 143.4434814
9: -49.4679642, 63.4686165, -49.4556770, 63.3622971, -112.8302612, 112.9242935
10: -77.0324554, 71.7843094, -76.8740234, 71.7302551, -148.7627106, 148.6583252
11: -81.3423843, 37.4320755, -80.9563980, 37.3852348, -118.7276154, 118.3884659
12: -84.9808960, 51.1354561, -84.9763336, 51.1430016, -136.1238708, 136.1117859
13: -77.4212875, 80.9588928, -77.3461304, 80.6051865, -158.0264740, 158.3050232
14: -117.8555450, 55.4265747, -117.5428085, 55.4611435, -173.3166809, 172.9693756
15: -60.4289513, 63.2446327, -60.2916107, 63.0661354, -123.4950867, 123.5362396
16: -79.4558411, 54.6678581, -79.3174057, 54.6475105, -134.1033325, 133.9852600
17: -111.2058792, 47.6980743, -110.6664886, 47.7092552, -158.9151306, 158.3645630
18: -79.2913971, 54.2288437, -78.8944321, 54.2145538, -133.5059357, 133.1232758
19: -58.0468903, 35.9874458, -57.5921631, 35.9340973, -93.9809875, 93.5796051
20: -56.5816917, 39.6843796, -56.3884964, 39.6213188, -96.2030029, 96.0728683
21: -74.4891663, 41.4193268, -74.0565948, 41.3814697, -115.8706360, 115.4759216
22: -69.1662598, 43.9837379, -68.9586639, 44.0488091, -113.2150726, 112.9423981
23: -61.8048019, 46.6068420, -61.3896866, 46.5320358, -108.3368301, 107.9965134
24: -73.5108795, 46.1007004, -73.1975403, 45.9934769, -119.5043564, 119.2982407
25: -64.2964096, 47.4006882, -63.9316483, 47.3371735, -111.6335831, 111.3323364
26: -83.2041473, 61.6542549, -82.8522186, 61.6690788, -144.8732300, 144.5064697
27: -69.3715820, 45.8917274, -69.1896057, 45.8518982, -115.2234802, 115.0813217
28: -58.4302826, 48.7245331, -58.1300545, 48.6334229, -107.0636978, 106.8545837
29: -75.3645630, 42.1570282, -74.9821320, 42.1720886, -117.5366364, 117.1391602
30: -79.2723083, 47.7642555, -78.7318115, 47.6452026, -126.9175110, 126.4960556
31: -80.5150299, 47.7197189, -79.9860229, 47.6819763, -128.1969910, 127.7057419
32: -83.3532562, 42.6687241, -83.3979187, 42.5654640, -125.9187088, 126.0666351
33: -109.6607971, 52.3655968, -109.6009827, 52.1027985, -161.7635803, 161.9665833
34: -97.6482773, 28.5688267, -97.5799866, 28.5017586, -126.1500244, 126.1487961
35: -91.3752441, 39.8269997, -91.3594971, 39.7109146, -131.0861511, 131.1864929
36: -89.8934860, 45.6374855, -89.8397141, 45.4634361, -135.3569183, 135.4772034
37: -131.2863159, 40.3470230, -131.1230011, 40.1955147, -171.4818268, 171.4700317
38: -106.6002884, 49.7973938, -106.5413895, 49.5351944, -156.1354828, 156.3387756
39: -118.4441452, 57.3015213, -118.3730774, 57.1094093, -175.5535583, 175.6745911
40: -100.0466995, 35.3931847, -99.9466095, 35.2026863, -135.2493896, 135.3397827
41: -84.0627365, 51.1663857, -83.9711990, 51.0036011, -135.0663452, 135.1375732
42: -66.1358032, 38.0078468, -66.1104889, 37.8350067, -103.9707947, 104.1183319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1036525, upper bound: 70.2575274
time: 79.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1036525, upper bound: 70.2576854
time: 149.54 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -91.4144440, 65.8197861, -91.4643173, 65.8128052, -157.2272491, 157.2840881
1: -45.6005287, 56.2928543, -45.6492004, 56.1381760, -101.7387085, 101.9420547
2: -39.8571625, 57.3505936, -39.9361458, 57.2261124, -97.0832748, 97.2867432
3: -49.8942604, 59.5194702, -49.9711685, 59.2472458, -109.1414871, 109.4906387
4: -48.7526855, 73.3724976, -48.8216820, 73.1660156, -121.9187012, 122.1941833
5: -46.1436310, 58.3315887, -46.2214241, 58.1389771, -104.2826080, 104.5530090
6: -90.9454880, 43.7392082, -91.0159302, 43.7565613, -134.7020569, 134.7551270
7: -54.8654366, 56.8247147, -54.9389610, 56.7225800, -111.5880127, 111.7636719
8: -60.6563568, 82.9627151, -60.7420731, 82.8290787, -143.4854279, 143.7047882
9: -49.4895401, 63.5900345, -49.6648254, 63.5944901, -113.0840302, 113.2548599
10: -77.0650177, 71.9251480, -77.1264038, 72.0044250, -149.0694427, 149.0515442
11: -81.4371414, 37.4546814, -81.1707764, 37.5546417, -118.9917755, 118.6254578
12: -85.0902176, 51.1654930, -85.1895370, 51.3075066, -136.3977203, 136.3550110
13: -77.4569092, 81.0450439, -77.5223846, 80.7974548, -158.2543640, 158.5674286
14: -117.9102707, 55.5463943, -117.7837677, 55.6880836, -173.5983429, 173.3301697
15: -60.4582367, 63.4062538, -60.5484428, 63.3565331, -123.8147736, 123.9546967
16: -79.4913635, 54.7166977, -79.4891129, 54.7684784, -134.2598419, 134.2057953
17: -111.2571106, 47.7392883, -110.8705139, 47.8299370, -159.0870361, 158.6098022
18: -79.3254547, 54.2596245, -79.0016022, 54.3193054, -133.6447449, 133.2612305
19: -58.1327629, 36.0048752, -57.7676353, 36.0640945, -94.1968536, 93.7725067
20: -56.6684990, 39.7017479, -56.5668106, 39.7708817, -96.4393692, 96.2685547
21: -74.5615387, 41.4504700, -74.2264252, 41.5437851, -116.1053162, 115.6768951
22: -69.2124176, 44.0013351, -69.0768738, 44.1261673, -113.3385773, 113.0782089
23: -61.8927994, 46.6320648, -61.5660820, 46.6778107, -108.5706100, 108.1981277
24: -73.6337585, 46.1177444, -73.4312744, 46.1483307, -119.7820892, 119.5490189
25: -64.4048157, 47.4230652, -64.1433411, 47.5068207, -111.9116135, 111.5664062
26: -83.2548676, 61.6829758, -82.9857941, 61.7932091, -145.0480652, 144.6687622
27: -69.4653015, 45.9080048, -69.3785858, 45.9569321, -115.4222260, 115.2865906
28: -58.5435410, 48.7411499, -58.3425560, 48.7826958, -107.3262253, 107.0837021
29: -75.4190979, 42.1687737, -75.1228409, 42.2481689, -117.6672668, 117.2916107
30: -79.4150009, 47.7909355, -79.0079880, 47.8564301, -127.2714310, 126.7989120
31: -80.6361237, 47.7443733, -80.2246017, 47.8312683, -128.4673767, 127.9689713
32: -83.4766922, 42.6875534, -83.6339874, 42.6992149, -126.1759033, 126.3215332
33: -109.7768860, 52.3855133, -109.8294678, 52.2995148, -162.0764008, 162.2149658
34: -97.7639465, 28.5861816, -97.8007278, 28.6676407, -126.4315872, 126.3869019
35: -91.4473114, 39.8412285, -91.5031738, 39.8557472, -131.3030548, 131.3443909
36: -89.9943161, 45.6517410, -90.0306473, 45.5857430, -135.5800476, 135.6823730
37: -131.4627991, 40.3694458, -131.4532318, 40.4150391, -171.8778076, 171.8226776
38: -106.7174377, 49.8158722, -106.7765045, 49.7002487, -156.4176636, 156.5923767
39: -118.5540466, 57.3224831, -118.5963669, 57.2616959, -175.8157349, 175.9188538
40: -100.1829147, 35.4099426, -100.2117844, 35.3526878, -135.5355988, 135.6217194
41: -84.1908264, 51.1864471, -84.2127991, 51.1590691, -135.3498840, 135.3992462
42: -66.2753830, 38.0277748, -66.3690109, 37.9978981, -104.2732697, 104.3967896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1409378, upper bound: 70.2575274
time: 80.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2576854
time: 92.03 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -91.3703995, 65.8184967, -91.0915833, 65.5545349, -156.9249115, 156.9100800
1: -45.5903130, 56.1817017, -45.4350281, 55.8854370, -101.4757538, 101.6167297
2: -39.8841095, 57.2243538, -39.7223816, 56.8785706, -96.7626724, 96.9467316
3: -49.9339981, 59.2672005, -49.7536049, 58.8656082, -108.7995911, 109.0208054
4: -48.7829971, 73.1344528, -48.5591469, 72.7078857, -121.4908829, 121.6935959
5: -46.2001305, 58.1543388, -46.0031624, 57.7875977, -103.9877167, 104.1575012
6: -90.7580414, 43.7668839, -90.4474182, 43.5129204, -134.2709503, 134.2142944
7: -54.8750381, 56.7816048, -54.7049789, 56.5240784, -111.3991165, 111.4865799
8: -60.6796799, 82.8144379, -60.4650002, 82.3907776, -143.0704651, 143.2794189
9: -49.5869408, 63.4827194, -49.2040176, 63.2334557, -112.8203964, 112.6867371
10: -76.9105377, 71.7798157, -76.2382660, 71.5204620, -148.4309998, 148.0180817
11: -80.9108353, 37.4328690, -80.3280182, 37.2508774, -118.1617050, 117.7608871
12: -85.0161972, 51.2020416, -84.4822769, 50.9544029, -135.9705963, 135.6843109
13: -77.4143753, 80.6633911, -77.2832642, 80.3906555, -157.8050232, 157.9466553
14: -117.5362091, 55.4312286, -116.9574585, 55.2571945, -172.7933960, 172.3886871
15: -60.4431572, 63.1179352, -60.1160507, 62.8066330, -123.2497864, 123.2339859
16: -79.2874756, 54.6984100, -78.8585205, 54.5174255, -133.8049011, 133.5569305
17: -110.7827530, 47.6886101, -110.3786240, 47.5441704, -158.3269196, 158.0672302
18: -78.9656830, 54.2292786, -78.6562653, 54.1434097, -133.1091003, 132.8855438
19: -57.7709541, 36.0003433, -57.4320374, 35.8941269, -93.6650848, 93.4323807
20: -56.4458122, 39.7005272, -56.1419144, 39.5495834, -95.9953918, 95.8424301
21: -74.1526337, 41.4351654, -73.6825104, 41.2905273, -115.4431610, 115.1176605
22: -69.0135040, 43.9926682, -68.7670898, 43.8784332, -112.8919373, 112.7597504
23: -61.5264511, 46.5956421, -61.2481537, 46.4788742, -108.0053253, 107.8437958
24: -73.2585602, 46.0821190, -73.0210190, 45.9761238, -119.2346802, 119.1031189
25: -64.0637054, 47.4224396, -63.8242645, 47.2530060, -111.3167114, 111.2467041
26: -82.9893799, 61.6911621, -82.6084595, 61.5584641, -144.5478516, 144.2996216
27: -69.2264709, 45.8691444, -68.9624863, 45.7910690, -115.0175400, 114.8316345
28: -58.2620010, 48.7479019, -58.0180244, 48.6222687, -106.8842621, 106.7659225
29: -75.0891266, 42.1685791, -74.7992706, 42.0874100, -117.1765289, 116.9678421
30: -78.9526291, 47.7701645, -78.5957794, 47.5802917, -126.5329208, 126.3659439
31: -80.1454926, 47.7214737, -79.7471695, 47.6000595, -127.7455521, 127.4686432
32: -83.4586639, 42.6929359, -83.1289520, 42.4901505, -125.9488068, 125.8218765
33: -109.6229248, 52.1781960, -109.4203796, 51.8128853, -161.4358063, 161.5985718
34: -97.6228333, 28.5073433, -97.4361877, 28.2495861, -125.8724213, 125.9435272
35: -91.3599091, 39.7236061, -91.2233124, 39.4411659, -130.8010712, 130.9469147
36: -89.8822937, 45.5408936, -89.7221985, 45.3616867, -135.2439728, 135.2630768
37: -131.2363739, 40.3717766, -130.9150696, 40.1225586, -171.3589325, 171.2868500
38: -106.5871887, 49.6461105, -106.3720245, 49.3674774, -155.9546661, 156.0181274
39: -118.4259186, 57.2065201, -118.2064285, 56.9510803, -175.3769989, 175.4129333
40: -100.0245590, 35.3044281, -99.7419586, 35.0787659, -135.1033173, 135.0463867
41: -84.0578003, 51.1254768, -83.8479614, 50.9403229, -134.9981232, 134.9734344
42: -66.1449432, 38.0386581, -65.8737488, 37.8037148, -103.9486542, 103.9123993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2373597
time: 88.87 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2375630
time: 75.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -91.4168777, 65.9161148, -91.3310471, 65.7399292, -157.1567993, 157.2471619
1: -45.6076431, 56.2390747, -45.5697708, 56.0004730, -101.6081085, 101.8088379
2: -39.9033813, 57.3114624, -39.8615913, 57.0365562, -96.9399414, 97.1730499
3: -49.9534721, 59.3797684, -49.9301147, 59.0737877, -109.0272522, 109.3098831
4: -48.8011627, 73.2637482, -48.7400131, 72.9432831, -121.7444305, 122.0037460
5: -46.2243385, 58.2656059, -46.1718979, 57.9908028, -104.2151413, 104.4375000
6: -90.9454422, 43.7845078, -90.7933655, 43.7037773, -134.6492157, 134.5778656
7: -54.8973503, 56.8366356, -54.8189316, 56.6332436, -111.5305939, 111.6555634
8: -60.6991615, 82.9135895, -60.6271477, 82.5803909, -143.2795563, 143.5407104
9: -49.6085892, 63.6041870, -49.4132538, 63.4656219, -113.0741959, 113.0174408
10: -76.9431458, 71.9206848, -76.4907837, 71.7947006, -148.7378540, 148.4114685
11: -81.0055771, 37.4554710, -80.5426788, 37.4202232, -118.4257965, 117.9981537
12: -85.1255035, 51.2320633, -84.6955261, 51.1186867, -136.2441864, 135.9275818
13: -77.4500122, 80.7495346, -77.4595108, 80.5831070, -158.0331116, 158.2090454
14: -117.5909271, 55.5511551, -117.1985321, 55.4842415, -173.0751648, 172.7496948
15: -60.4724159, 63.2795334, -60.3727531, 63.0971107, -123.5695267, 123.6522827
16: -79.3230362, 54.7472153, -79.0303802, 54.6384277, -133.9614563, 133.7775879
17: -110.8341675, 47.7299500, -110.5829849, 47.6646347, -158.4988098, 158.3129272
18: -78.9998474, 54.2600632, -78.7633972, 54.2480812, -133.2478943, 133.0234680
19: -57.8568535, 36.0177727, -57.6075706, 36.0239944, -93.8808441, 93.6253433
20: -56.5326271, 39.7178726, -56.3203011, 39.6990662, -96.2316895, 96.0381622
21: -74.2250214, 41.4662628, -73.8525238, 41.4525986, -115.6776123, 115.3187866
22: -69.0597382, 44.0102768, -68.8852997, 43.9557571, -113.0154724, 112.8955765
23: -61.6144409, 46.6208267, -61.4245911, 46.6245461, -108.2389832, 108.0454102
24: -73.3814697, 46.0992050, -73.2546997, 46.1310043, -119.5124741, 119.3539047
25: -64.1721115, 47.4448166, -64.0358810, 47.4224854, -111.5945892, 111.4806976
26: -83.0400696, 61.7198029, -82.7421112, 61.6824570, -144.7225342, 144.4619141
27: -69.3203125, 45.8854370, -69.1516113, 45.8959999, -115.2163086, 115.0370407
28: -58.3753242, 48.7645149, -58.2304993, 48.7715912, -107.1468964, 106.9950104
29: -75.1436310, 42.1803436, -74.9400330, 42.1635284, -117.3071518, 117.1203766
30: -79.0952301, 47.7967644, -78.8720322, 47.7914314, -126.8866577, 126.6687927
31: -80.2666168, 47.7461243, -79.9858932, 47.7492714, -128.0158691, 127.7320175
32: -83.5821075, 42.7117805, -83.3651276, 42.6238327, -126.2059326, 126.0769043
33: -109.7390213, 52.1981277, -109.6488190, 52.0095978, -161.7485962, 161.8469391
34: -97.7385406, 28.5246544, -97.6569901, 28.4154510, -126.1539917, 126.1816406
35: -91.4320068, 39.7378845, -91.3670044, 39.5860443, -131.0180511, 131.1048889
36: -89.9831543, 45.5551109, -89.9132385, 45.4839935, -135.4671326, 135.4683533
37: -131.4129639, 40.3941917, -131.2453308, 40.3420486, -171.7550049, 171.6395111
38: -106.7043686, 49.6645699, -106.6071548, 49.5324669, -156.2368317, 156.2717285
39: -118.5357819, 57.2275543, -118.4296799, 57.1033173, -175.6390991, 175.6572266
40: -100.1607513, 35.3211594, -100.0075302, 35.2286797, -135.3894348, 135.3286896
41: -84.1858826, 51.1454697, -84.0895691, 51.0957870, -135.2816772, 135.2350311
42: -66.2845917, 38.0585670, -66.1323776, 37.9665985, -104.2511749, 104.1909256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1625

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2373597
time: 90.92 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2375630
time: 89.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -91.5806122, 65.9241791, -91.0707855, 65.5442505, -157.1248627, 156.9949646
1: -45.7293625, 56.4886513, -45.4247398, 55.8816528, -101.6110153, 101.9133911
2: -40.0306778, 57.5551720, -39.7157898, 56.8733444, -96.9040222, 97.2709656
3: -50.0804825, 59.7338753, -49.7461739, 58.8578835, -108.9383545, 109.4800491
4: -48.9745789, 73.5726013, -48.5520744, 72.7022247, -121.6768036, 122.1246643
5: -46.3593407, 58.5089073, -45.9964638, 57.7813416, -104.1406860, 104.5053635
6: -90.9352951, 43.8219261, -90.4368439, 43.4630432, -134.3983307, 134.2587738
7: -55.0674477, 56.9827919, -54.6969681, 56.5192337, -111.5866852, 111.6797638
8: -60.8331223, 83.2136993, -60.4585495, 82.3825912, -143.2157135, 143.6722260
9: -49.7059212, 63.6303215, -49.1889954, 63.2226639, -112.9285889, 112.8193207
10: -77.5314484, 72.0487518, -76.2314148, 71.5094147, -149.0408630, 148.2801666
11: -81.8303528, 37.6113472, -80.3148880, 37.2423706, -119.0727158, 117.9262238
12: -85.4480438, 51.4077377, -84.4759979, 50.9459229, -136.3939667, 135.8837280
13: -77.5370026, 81.1248322, -77.2493439, 80.3768616, -157.9138489, 158.3741760
14: -118.3255844, 55.6359863, -116.9447098, 55.2494278, -173.5750122, 172.5806885
15: -60.6019974, 63.4541206, -60.0791626, 62.7981071, -123.4001007, 123.5332794
16: -79.8396683, 54.8487930, -78.8407669, 54.4734192, -134.3130798, 133.6895599
17: -111.5072250, 47.8674431, -110.3707275, 47.5329514, -159.0401764, 158.2381744
18: -79.5759583, 54.3759766, -78.6445541, 54.1233521, -133.6993103, 133.0205383
19: -58.3303299, 36.1297531, -57.4258347, 35.8889885, -94.2193146, 93.5555878
20: -56.8254433, 39.8135643, -56.1364479, 39.5410309, -96.3664703, 95.9500046
21: -74.8898315, 41.6031799, -73.6753693, 41.2855682, -116.1753998, 115.2785416
22: -69.3198471, 44.1134300, -68.7613220, 43.8650742, -113.1849213, 112.8747559
23: -62.0228119, 46.7412529, -61.2423172, 46.4719276, -108.4947357, 107.9835663
24: -73.6011353, 46.1616554, -73.0073395, 45.9574890, -119.5586243, 119.1689835
25: -64.4460144, 47.5494308, -63.8140450, 47.2451019, -111.6911163, 111.3634720
26: -83.5462952, 61.9203529, -82.5981598, 61.5490227, -145.0953064, 144.5185089
27: -69.5017014, 45.9438400, -68.9515076, 45.7803879, -115.2820892, 114.8953476
28: -58.5815125, 48.8661919, -58.0132256, 48.6144447, -107.1959534, 106.8794174
29: -75.5559387, 42.2819595, -74.7925568, 42.0800209, -117.6359558, 117.0745163
30: -79.5117188, 47.9394989, -78.5830078, 47.5714607, -127.0831757, 126.5225067
31: -80.8419189, 47.8685112, -79.7388763, 47.5931702, -128.4350891, 127.6073837
32: -83.5874252, 42.7921219, -83.1077728, 42.4687729, -126.0561829, 125.8998947
33: -109.8005066, 52.5917625, -109.4075012, 51.8072166, -161.6077118, 161.9992676
34: -97.7556686, 28.7236462, -97.4265289, 28.2403831, -125.9960480, 126.1501770
35: -91.4548950, 40.0072746, -91.2035980, 39.4375534, -130.8924561, 131.2108765
36: -89.9991379, 45.7423859, -89.7038422, 45.3581734, -135.3572998, 135.4462128
37: -131.4404907, 40.4871902, -130.8924255, 40.0931625, -171.5336304, 171.3796082
38: -106.7857437, 49.9271927, -106.3577118, 49.3609428, -156.1466827, 156.2849121
39: -118.6204758, 57.4376068, -118.1857452, 56.9466667, -175.5671387, 175.6233521
40: -100.2262039, 35.4949837, -99.7286301, 35.0731773, -135.2993774, 135.2236176
41: -84.2016907, 51.2573471, -83.8312836, 50.9226875, -135.1243744, 135.0886230
42: -66.3002396, 38.1657028, -65.8685989, 37.7599640, -104.0601959, 104.0343018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1814861, upper bound: 70.2372314
time: 101.49 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2374416
time: 147.91 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -91.6272507, 66.0218048, -91.3102570, 65.7296448, -157.3568878, 157.3320618
1: -45.7467461, 56.5460281, -45.5594940, 55.9966736, -101.7434235, 102.1055222
2: -40.0500336, 57.6422081, -39.8549767, 57.0313492, -97.0813828, 97.4971848
3: -50.0999985, 59.8464737, -49.9226913, 59.0660591, -109.1660614, 109.7691650
4: -48.9927979, 73.7018433, -48.7329445, 72.9376678, -121.9304657, 122.4347839
5: -46.3836365, 58.6201630, -46.1652527, 57.9845467, -104.3681793, 104.7854156
6: -91.1225281, 43.8395920, -90.7827911, 43.6538887, -134.7764130, 134.6223755
7: -55.0899010, 57.0377846, -54.8108940, 56.6283836, -111.7182770, 111.8486786
8: -60.8526649, 83.3127060, -60.6206932, 82.5722961, -143.4249573, 143.9333801
9: -49.7275391, 63.7517166, -49.3982239, 63.4548874, -113.1824265, 113.1499405
10: -77.5639725, 72.1896133, -76.4839706, 71.7837219, -149.3476868, 148.6735840
11: -81.9250793, 37.6339531, -80.5296021, 37.4117050, -119.3367844, 118.1635590
12: -85.5573578, 51.4378929, -84.6892929, 51.1102371, -136.6676025, 136.1271820
13: -77.5726242, 81.2109070, -77.4256287, 80.5692673, -158.1418915, 158.6365356
14: -118.3802643, 55.7557983, -117.1858215, 55.4764938, -173.8567352, 172.9416199
15: -60.6313629, 63.6157570, -60.3358154, 63.0885582, -123.7199249, 123.9515686
16: -79.8751831, 54.8975906, -79.0125809, 54.5944290, -134.4696045, 133.9101715
17: -111.5583801, 47.9086914, -110.5750504, 47.6534348, -159.2118225, 158.4837341
18: -79.6100082, 54.4068069, -78.7516937, 54.2280540, -133.8380585, 133.1585083
19: -58.4161682, 36.1472054, -57.6014099, 36.0188904, -94.4350586, 93.7486115
20: -56.9122353, 39.8309402, -56.3148117, 39.6905136, -96.6027451, 96.1457520
21: -74.9621124, 41.6343651, -73.8453979, 41.4476471, -116.4097595, 115.4797668
22: -69.3659897, 44.1310806, -68.8795319, 43.9424286, -113.3084106, 113.0106049
23: -62.1107483, 46.7664795, -61.4187698, 46.6175461, -108.7282944, 108.1852493
24: -73.7239990, 46.1787720, -73.2410126, 46.1123199, -119.8363190, 119.4197845
25: -64.5543900, 47.5718575, -64.0256500, 47.4145508, -111.9689407, 111.5975037
26: -83.5969238, 61.9491425, -82.7317810, 61.6730804, -145.2700043, 144.6809235
27: -69.5953674, 45.9601364, -69.1406479, 45.8853798, -115.4807434, 115.1007767
28: -58.6947136, 48.8827972, -58.2256927, 48.7637558, -107.4584656, 107.1084747
29: -75.6104050, 42.2936516, -74.9333038, 42.1561546, -117.7665558, 117.2269592
30: -79.6543045, 47.9661751, -78.8592224, 47.7825623, -127.4368668, 126.8253937
31: -80.9630127, 47.8932037, -79.9775772, 47.7424088, -128.7054138, 127.8707733
32: -83.7108154, 42.8110046, -83.3439331, 42.6024323, -126.3132324, 126.1549377
33: -109.9165955, 52.6116333, -109.6359253, 52.0038872, -161.9204712, 162.2475586
34: -97.8714066, 28.7409782, -97.6473389, 28.4062119, -126.2776184, 126.3883209
35: -91.5269928, 40.0215874, -91.3473358, 39.5824013, -131.1093903, 131.3689270
36: -90.0999756, 45.7567062, -89.8948364, 45.4804459, -135.5803986, 135.6515503
37: -131.6169739, 40.5095978, -131.2226105, 40.3126144, -171.9295807, 171.7322083
38: -106.9029465, 49.9457016, -106.5928726, 49.5260048, -156.4289551, 156.5385742
39: -118.7303314, 57.4586449, -118.4090500, 57.0988464, -175.8291779, 175.8676910
40: -100.3621292, 35.5117798, -99.9941559, 35.2230682, -135.5851746, 135.5059357
41: -84.3297729, 51.2774048, -84.0729446, 51.0782242, -135.4079895, 135.3503418
42: -66.4397583, 38.1856766, -66.1272278, 37.9228287, -104.3625870, 104.3128967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2372314
time: 83.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2374416
time: 87.02 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -91.3970718, 65.8159943, -91.3158264, 65.6561279, -157.0531921, 157.1318207
1: -45.6054611, 56.1890182, -45.5758018, 56.0376320, -101.6430893, 101.7648163
2: -39.9163780, 57.2280502, -39.8874168, 57.0825653, -96.9989319, 97.1154633
3: -49.9589348, 59.2767639, -49.8963661, 59.0626106, -109.0215454, 109.1731262
4: -48.8019447, 73.1407852, -48.7400284, 72.9476852, -121.7496262, 121.8808060
5: -46.2265854, 58.1668282, -46.1632996, 57.9572334, -104.1838226, 104.3301163
6: -90.7726135, 43.7606239, -90.7074432, 43.6206245, -134.3932343, 134.4680634
7: -54.8901901, 56.7894478, -54.9011307, 56.6291275, -111.5193176, 111.6905823
8: -60.7207451, 82.8263321, -60.6715775, 82.6631165, -143.3838501, 143.4978943
9: -49.5951004, 63.5331841, -49.4915581, 63.4400635, -113.0351639, 113.0247421
10: -76.9288635, 71.8831177, -76.9125671, 71.8799744, -148.8088379, 148.7956848
11: -80.9327316, 37.5017929, -80.9961166, 37.4913025, -118.4240265, 118.4978943
12: -85.0266876, 51.2939224, -85.0006866, 51.2950974, -136.3217773, 136.2946167
13: -77.4188461, 80.7030106, -77.4042358, 80.6595459, -158.0783997, 158.1072388
14: -117.5481186, 55.5366936, -117.5902405, 55.5949898, -173.1431122, 173.1269379
15: -60.4987297, 63.1335182, -60.3881531, 63.0986595, -123.5973892, 123.5216675
16: -79.3034821, 54.7634354, -79.3698654, 54.7696381, -134.0731201, 134.1333008
17: -110.7898102, 47.7202911, -110.6948776, 47.7762566, -158.5660553, 158.4151611
18: -78.9727936, 54.2507896, -78.9321594, 54.2906876, -133.2634888, 133.1829376
19: -57.7823639, 36.0163040, -57.6224823, 35.9996490, -93.7820129, 93.6387787
20: -56.4636726, 39.7402573, -56.4193420, 39.6962891, -96.1599579, 96.1595993
21: -74.1658096, 41.4793968, -74.0861816, 41.4728317, -115.6386337, 115.5655670
22: -69.0311432, 44.0064926, -68.9845581, 44.0885735, -113.1197205, 112.9910507
23: -61.5368462, 46.6151047, -61.4144020, 46.5920792, -108.1289215, 108.0294952
24: -73.2980194, 46.0809669, -73.2318420, 46.0246201, -119.3226395, 119.3128052
25: -64.0717773, 47.4461288, -63.9586372, 47.4001312, -111.4719009, 111.4047623
26: -83.0073853, 61.7445221, -82.8881149, 61.8041458, -144.8115234, 144.6326141
27: -69.2827148, 45.8742447, -69.2409668, 45.8740768, -115.1567841, 115.1152039
28: -58.2765274, 48.7515068, -58.1558647, 48.6940002, -106.9705276, 106.9073715
29: -75.1067047, 42.1790733, -75.0033569, 42.2154007, -117.3221054, 117.1824265
30: -78.9699783, 47.7879372, -78.7643433, 47.7190590, -126.6890411, 126.5522690
31: -80.1657104, 47.7615662, -80.0263672, 47.7612228, -127.9269333, 127.7879333
32: -83.4754181, 42.7289276, -83.4395599, 42.6404190, -126.1158218, 126.1684875
33: -109.6828384, 52.1923332, -109.6776581, 52.1338348, -161.8166656, 161.8699951
34: -97.6632080, 28.5173340, -97.6309052, 28.5333385, -126.1965332, 126.1482162
35: -91.4052887, 39.7370720, -91.4181213, 39.7347069, -131.1399841, 131.1551971
36: -89.8986740, 45.5484352, -89.8796387, 45.4846001, -135.3832703, 135.4280701
37: -131.2698669, 40.3847237, -131.1737061, 40.2620392, -171.5319061, 171.5584259
38: -106.6018753, 49.6649055, -106.5964127, 49.5622025, -156.1640778, 156.2613220
39: -118.4444733, 57.2414207, -118.4337616, 57.1493912, -175.5938263, 175.6751709
40: -100.0510254, 35.3095627, -99.9925766, 35.2164345, -135.2674561, 135.3021240
41: -84.0705414, 51.1305389, -84.0155945, 51.0335274, -135.1040497, 135.1461334
42: -66.1591949, 38.0198288, -66.1363068, 37.8957291, -104.0549164, 104.1561356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 789

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2577376
time: 122.28 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1776395, upper bound: 70.2579013
time: 99.99 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -91.4435501, 65.9135895, -91.5555038, 65.8415146, -157.2850647, 157.4690857
1: -45.6227913, 56.2463989, -45.7105904, 56.1526489, -101.7754364, 101.9569855
2: -39.9356499, 57.3151550, -40.0266647, 57.2405548, -97.1762085, 97.3418121
3: -49.9784012, 59.3893356, -50.0729561, 59.2707176, -109.2491150, 109.4622879
4: -48.8201027, 73.2700958, -48.9209824, 73.1830978, -122.0031967, 122.1910706
5: -46.2507858, 58.2781181, -46.3321877, 58.1603737, -104.4111633, 104.6102982
6: -90.9599915, 43.7782288, -91.0532379, 43.8115044, -134.7714691, 134.8314514
7: -54.9125061, 56.8444824, -55.0151825, 56.7382736, -111.6507797, 111.8596649
8: -60.7402573, 82.9254379, -60.8338356, 82.8527069, -143.5929565, 143.7592773
9: -49.6167297, 63.6546326, -49.7007599, 63.6722412, -113.2889709, 113.3553925
10: -76.9614410, 72.0239868, -77.1650467, 72.1541595, -149.1156006, 149.1890259
11: -81.0274963, 37.5244026, -81.2104416, 37.6606712, -118.6881714, 118.7348480
12: -85.1359863, 51.3239822, -85.2138672, 51.4595833, -136.5955505, 136.5378418
13: -77.4544525, 80.7892151, -77.5804977, 80.8518524, -158.3062897, 158.3697205
14: -117.6028671, 55.6565857, -117.8312531, 55.8219910, -173.4248657, 173.4878235
15: -60.5279770, 63.2951126, -60.6449585, 63.3890114, -123.9169769, 123.9400711
16: -79.3390732, 54.8122292, -79.5416336, 54.8906479, -134.2297211, 134.3538513
17: -110.8413239, 47.7615967, -110.8989639, 47.8969612, -158.7382812, 158.6605530
18: -79.0069275, 54.2815971, -79.0393372, 54.3954315, -133.4023285, 133.3209229
19: -57.8682251, 36.0337448, -57.7979622, 36.1296310, -93.9978561, 93.8317032
20: -56.5504761, 39.7576065, -56.5976372, 39.8458328, -96.3963089, 96.3552399
21: -74.2381897, 41.5105095, -74.2560120, 41.6351166, -115.8733063, 115.7665253
22: -69.0773926, 44.0241547, -69.1028061, 44.1659317, -113.2433167, 113.1269608
23: -61.6248322, 46.6402893, -61.5908318, 46.7378349, -108.3626709, 108.2311249
24: -73.4209442, 46.0980453, -73.4655457, 46.1795082, -119.6004410, 119.5635910
25: -64.1801605, 47.4684753, -64.1703186, 47.5697632, -111.7499237, 111.6387939
26: -83.0580750, 61.7732315, -83.0217133, 61.9282837, -144.9863586, 144.7949524
27: -69.3765106, 45.8905296, -69.4299240, 45.9790497, -115.3555603, 115.3204498
28: -58.3898277, 48.7680817, -58.3683510, 48.8433495, -107.2331696, 107.1364212
29: -75.1612320, 42.1908264, -75.1441040, 42.2915039, -117.4527359, 117.3349304
30: -79.1126251, 47.8145752, -79.0404968, 47.9302711, -127.0428925, 126.8550720
31: -80.2868042, 47.7862358, -80.2649841, 47.9105606, -128.1973572, 128.0512238
32: -83.5988007, 42.7477875, -83.6755981, 42.7741966, -126.3729935, 126.4233780
33: -109.7989655, 52.2122574, -109.9061432, 52.3305054, -162.1294708, 162.1184082
34: -97.7789459, 28.5346527, -97.8516312, 28.6991653, -126.4781113, 126.3862839
35: -91.4773865, 39.7513428, -91.5618286, 39.8795471, -131.3569183, 131.3131714
36: -89.9995270, 45.5626831, -90.0705795, 45.6069260, -135.6064453, 135.6332703
37: -131.4464417, 40.4070969, -131.5040283, 40.4815903, -171.9280396, 171.9111176
38: -106.7190704, 49.6834869, -106.8314896, 49.7272415, -156.4463043, 156.5149841
39: -118.5543747, 57.2624588, -118.6569977, 57.3017464, -175.8561096, 175.9194641
40: -100.1871567, 35.3263168, -100.2578201, 35.3664093, -135.5535583, 135.5841370
41: -84.1986542, 51.1505203, -84.2571411, 51.1890411, -135.3876953, 135.4076538
42: -66.2988434, 38.0397263, -66.3948517, 38.0587006, -104.3575439, 104.4345779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 955

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2577376
time: 90.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1409378, upper bound: 70.2579013
time: 93.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -91.6080170, 65.9215851, -91.2950439, 65.6458511, -157.2538757, 157.2166290
1: -45.7449570, 56.4959869, -45.5654755, 56.0338745, -101.7788315, 102.0614624
2: -40.0640411, 57.5588760, -39.8807907, 57.0773582, -97.1413803, 97.4396591
3: -50.1058540, 59.7435112, -49.8889503, 59.0548515, -109.1607056, 109.6324615
4: -48.9937630, 73.5789642, -48.7328186, 72.9419861, -121.9357452, 122.3117752
5: -46.3862801, 58.5214386, -46.1566086, 57.9509964, -104.3372650, 104.6780396
6: -90.9500427, 43.8156433, -90.6969528, 43.5705986, -134.5206451, 134.5125732
7: -55.0834808, 56.9907150, -54.8929825, 56.6242790, -111.7077408, 111.8836975
8: -60.8748093, 83.2255249, -60.6651115, 82.6550369, -143.5298462, 143.8906403
9: -49.7142563, 63.6813049, -49.4766159, 63.4291687, -113.1434174, 113.1579208
10: -77.5497284, 72.1529999, -76.9058228, 71.8688660, -149.4185944, 149.0588226
11: -81.8522491, 37.6807632, -80.9830399, 37.4827805, -119.3350296, 118.6638031
12: -85.4584656, 51.5000916, -84.9944611, 51.2865715, -136.7450256, 136.4945374
13: -77.5417099, 81.1638947, -77.3701859, 80.6454086, -158.1871185, 158.5340881
14: -118.3376541, 55.7415543, -117.5774689, 55.5871429, -173.9247894, 173.3190308
15: -60.6582108, 63.4697800, -60.3512192, 63.0904884, -123.7486877, 123.8209915
16: -79.8556290, 54.9146614, -79.3524857, 54.7256279, -134.5812378, 134.2671509
17: -111.5145340, 47.8993912, -110.6868896, 47.7649078, -159.2794495, 158.5862732
18: -79.5829468, 54.3977356, -78.9203033, 54.2706032, -133.8535461, 133.3180389
19: -58.3424454, 36.1461029, -57.6162872, 35.9945145, -94.3369446, 93.7623901
20: -56.8432922, 39.8537216, -56.4138794, 39.6876640, -96.5309448, 96.2676010
21: -74.9030609, 41.6480408, -74.0791016, 41.4678841, -116.3709412, 115.7271347
22: -69.3375244, 44.1278839, -68.9787445, 44.0752335, -113.4127579, 113.1066284
23: -62.0331993, 46.7612000, -61.4085579, 46.5851212, -108.6183090, 108.1697540
24: -73.6404953, 46.1605949, -73.2180176, 46.0060196, -119.6464920, 119.3786087
25: -64.4536972, 47.5735474, -63.9483185, 47.3921089, -111.8458099, 111.5218658
26: -83.5643997, 61.9741516, -82.8777618, 61.7946358, -145.3590393, 144.8519135
27: -69.5576782, 45.9490242, -69.2298813, 45.8634109, -115.4210815, 115.1789093
28: -58.5960884, 48.8699417, -58.1510315, 48.6862030, -107.2822876, 107.0209732
29: -75.5733795, 42.2928619, -74.9966354, 42.2080994, -117.7814789, 117.2894897
30: -79.5290451, 47.9572868, -78.7514877, 47.7101250, -127.2391663, 126.7087708
31: -80.8621368, 47.9091721, -80.0181580, 47.7544060, -128.6165466, 127.9273224
32: -83.6042938, 42.8285637, -83.4184265, 42.6186867, -126.2229767, 126.2469940
33: -109.8604431, 52.6059875, -109.6647491, 52.1281433, -161.9885864, 162.2707367
34: -97.7961731, 28.7338276, -97.6211548, 28.5242348, -126.3204041, 126.3549728
35: -91.5003662, 40.0208931, -91.3983994, 39.7312355, -131.2315979, 131.4192810
36: -90.0156860, 45.7500610, -89.8611603, 45.4809914, -135.4966736, 135.6112213
37: -131.4736328, 40.5002060, -131.1511230, 40.2327652, -171.7063904, 171.6513214
38: -106.8008957, 49.9459229, -106.5820160, 49.5556870, -156.3565674, 156.5279236
39: -118.6392899, 57.4725685, -118.4129486, 57.1448746, -175.7841644, 175.8855133
40: -100.2531509, 35.5000381, -99.9792786, 35.2108002, -135.4639587, 135.4793091
41: -84.2151871, 51.2624016, -83.9984283, 51.0159264, -135.2311096, 135.2608337
42: -66.3147049, 38.1450424, -66.1311569, 37.8518486, -104.1665497, 104.2761993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2575274
time: 84.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2205967, upper bound: 70.2576854
time: 85.54 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -91.6546173, 66.0191956, -91.5346985, 65.8312302, -157.4858398, 157.5538940
1: -45.7623520, 56.5534058, -45.7002449, 56.1488800, -101.9112320, 102.2536469
2: -40.0833588, 57.6458893, -40.0200539, 57.2353287, -97.3186798, 97.6659393
3: -50.1253433, 59.8560982, -50.0654984, 59.2629433, -109.3882828, 109.9216003
4: -49.0119896, 73.7081757, -48.9138107, 73.1774139, -122.1893921, 122.6219864
5: -46.4105911, 58.6327171, -46.3254890, 58.1541405, -104.5647278, 104.9581909
6: -91.1373291, 43.8333054, -91.0427094, 43.7614822, -134.8988037, 134.8760071
7: -55.1058464, 57.0457497, -55.0070229, 56.7333984, -111.8392487, 112.0527725
8: -60.8943748, 83.3245544, -60.8272972, 82.8445587, -143.7389374, 144.1518402
9: -49.7358818, 63.8027115, -49.6858101, 63.6613922, -113.3972702, 113.4885254
10: -77.5823059, 72.2938538, -77.1582870, 72.1430511, -149.7253571, 149.4521484
11: -81.9469452, 37.7033920, -81.1974106, 37.6521683, -119.5991135, 118.9007950
12: -85.5677567, 51.5301971, -85.2076035, 51.4510155, -137.0187683, 136.7377930
13: -77.5773468, 81.2500153, -77.5464706, 80.8377075, -158.4150543, 158.7964783
14: -118.3923798, 55.8614120, -117.8184509, 55.8140907, -174.2064667, 173.6798553
15: -60.6875610, 63.6313515, -60.6080093, 63.3808479, -124.0684052, 124.2393570
16: -79.8911743, 54.9634132, -79.5242310, 54.8466187, -134.7377930, 134.4876404
17: -111.5657043, 47.9406815, -110.8909302, 47.8856277, -159.4512939, 158.8316040
18: -79.6170197, 54.4285431, -79.0274887, 54.3753586, -133.9923706, 133.4560242
19: -58.4282265, 36.1635590, -57.7917747, 36.1245422, -94.5527649, 93.9553223
20: -56.9300652, 39.8710976, -56.5921669, 39.8372269, -96.7672806, 96.4632645
21: -74.9753723, 41.6792374, -74.2489014, 41.6301727, -116.6055450, 115.9281387
22: -69.3836517, 44.1455307, -69.0970078, 44.1525879, -113.5362396, 113.2425385
23: -62.1211586, 46.7864799, -61.5849571, 46.7308655, -108.8520203, 108.3714371
24: -73.7633514, 46.1776581, -73.4517365, 46.1608620, -119.9242096, 119.6293945
25: -64.5620728, 47.5959473, -64.1600037, 47.5617065, -112.1237793, 111.7559509
26: -83.6150970, 62.0028687, -83.0113449, 61.9188271, -145.5339050, 145.0142059
27: -69.6513519, 45.9653358, -69.4188538, 45.9684067, -115.6197510, 115.3841782
28: -58.7092896, 48.8865395, -58.3635139, 48.8355293, -107.5448151, 107.2500458
29: -75.6279144, 42.3045959, -75.1373596, 42.2842598, -117.9121704, 117.4419556
30: -79.6716156, 47.9840469, -79.0276260, 47.9213257, -127.5929413, 127.0116730
31: -80.9832077, 47.9338417, -80.2567368, 47.9036942, -128.8869019, 128.1905823
32: -83.7276611, 42.8475113, -83.6544495, 42.7524338, -126.4800949, 126.5019608
33: -109.9765091, 52.6258888, -109.8932419, 52.3247757, -162.3012848, 162.5191345
34: -97.9118423, 28.7511673, -97.8419189, 28.6900940, -126.6019135, 126.5930862
35: -91.5724869, 40.0351830, -91.5421448, 39.8761330, -131.4486237, 131.5773315
36: -90.1165009, 45.7643623, -90.0521088, 45.6033630, -135.7198639, 135.8164673
37: -131.6501465, 40.5226669, -131.4813385, 40.4522858, -172.1024323, 172.0039978
38: -106.9180603, 49.9644051, -106.8171158, 49.7207527, -156.6388092, 156.7815247
39: -118.7491913, 57.4936218, -118.6362076, 57.2972488, -176.0464478, 176.1298218
40: -100.3891220, 35.5168076, -100.2445068, 35.3608017, -135.7499237, 135.7613220
41: -84.3432312, 51.2824402, -84.2399750, 51.1714363, -135.5146637, 135.5224152
42: -66.4541779, 38.1649818, -66.3896942, 38.0147667, -104.4689484, 104.5546722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=455, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1748
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 969
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1004
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 968
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1526
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1700
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1625
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 789

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1758

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2575274
time: 75.57 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.2576850, upper bound: 70.2576854
time: 94.28 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 172.27 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2373597
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2375630
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2373597
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2375630
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2372314
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2374416
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2372314
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2374416
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1036525, upper bound: 70.2577376
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1036525, upper bound: 70.2579013
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1409378, upper bound: 70.2577376
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1409378, upper bound: 70.2579013
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1036525, upper bound: 70.2575274
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1036525, upper bound: 70.2576854
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1409378, upper bound: 70.2575274
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2576854
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2373597
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2375630
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2373597
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2375630
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1814861, upper bound: 70.2372314
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2374416
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2372314
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2374416
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2577376
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1776395, upper bound: 70.2579013
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2577376
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1409378, upper bound: 70.2579013
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2575274
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.2205967, upper bound: 70.2576854
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2575274
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 172.27
Output dim: 4, lower bound: -70.2576850, upper bound: 70.2576854

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -90.7439423, 65.4386215, -90.8154678, 65.5100861, -156.2540283, 156.2540894
1: -45.1431923, 55.8054466, -45.2305527, 55.8605194, -101.0037079, 101.0359955
2: -39.4031906, 56.8212395, -39.5015526, 56.8528137, -96.2560043, 96.3227921
3: -49.5023956, 58.8296623, -49.5584412, 58.8299561, -108.3323517, 108.3881073
4: -48.1806831, 72.6681824, -48.2848358, 72.6823425, -120.8630219, 120.9530106
5: -45.7272034, 57.7295990, -45.7922592, 57.7492065, -103.4764099, 103.5218582
6: -90.4568558, 43.5005493, -90.3992462, 43.4217453, -133.8786011, 133.8997803
7: -54.2792435, 56.4364624, -54.4450073, 56.4958191, -110.7750626, 110.8814697
8: -60.0356331, 82.2923584, -60.1628036, 82.3539124, -142.3895416, 142.4551544
9: -49.0607491, 63.1379280, -49.0354080, 63.1455765, -112.2063217, 112.1733398
10: -76.1214752, 71.2595825, -76.0624847, 71.3443146, -147.4657898, 147.3220673
11: -80.3053741, 37.0860481, -80.2576599, 37.1077957, -117.4131699, 117.3436966
12: -84.4075241, 50.5519867, -84.4378357, 50.6638451, -135.0713654, 134.9898224
13: -77.1820526, 80.3552628, -77.2049332, 80.3158493, -157.4978943, 157.5601959
14: -116.7597275, 54.9909897, -116.7625275, 55.1065369, -171.8662567, 171.7535095
15: -60.0866699, 62.8214188, -59.9951096, 62.7656555, -122.8523254, 122.8165207
16: -78.6185913, 54.3196793, -78.6678772, 54.4087753, -133.0273590, 132.9875488
17: -110.2720947, 47.3955688, -110.2650375, 47.4558792, -157.7279663, 157.6605988
18: -78.5535583, 53.9368362, -78.5904846, 54.0260963, -132.5796356, 132.5273132
19: -57.3723793, 35.7178307, -57.3788071, 35.7680397, -93.1404190, 93.0966339
20: -56.0895386, 39.4431305, -56.0874214, 39.4356537, -95.5251770, 95.5305481
21: -73.6125031, 41.1071472, -73.6159134, 41.1529427, -114.7654419, 114.7230453
22: -68.6973877, 43.6565018, -68.7069702, 43.7478256, -112.4452057, 112.3634720
23: -61.2130089, 46.2981262, -61.2077827, 46.3510437, -107.5640564, 107.5059052
24: -73.0400009, 45.9575043, -72.9754028, 45.9301872, -118.9701843, 118.9329071
25: -63.7830048, 47.0514107, -63.7775269, 47.0926208, -110.8756256, 110.8289337
26: -82.4819336, 61.2320671, -82.5384750, 61.3633881, -143.8453064, 143.7705383
27: -68.9179840, 45.7192192, -68.8890686, 45.7371559, -114.6551361, 114.6082840
28: -57.9797363, 48.4077606, -57.9698792, 48.4664154, -106.4461517, 106.3776398
29: -74.7513428, 41.8420029, -74.7467957, 41.9467239, -116.6980591, 116.5887985
30: -78.6024628, 47.4702377, -78.5399780, 47.4601936, -126.0626373, 126.0102005
31: -79.6676331, 47.3497047, -79.6815186, 47.4315033, -127.0991287, 127.0312195
32: -83.0841599, 42.3830261, -83.0809555, 42.3578148, -125.4419556, 125.4639740
33: -109.2486572, 51.7350960, -109.3058624, 51.6794357, -160.9280701, 161.0409546
34: -97.3511887, 28.1334209, -97.3658905, 28.1129608, -125.4641495, 125.4993057
35: -91.0862579, 39.3018341, -91.1489944, 39.2962952, -130.3825531, 130.4508362
36: -89.6139832, 45.1899109, -89.6745453, 45.2167549, -134.8307190, 134.8644562
37: -130.8667908, 40.0955544, -130.8320618, 40.0210419, -170.8878326, 170.9276123
38: -106.1859207, 49.3016052, -106.2852631, 49.2439384, -155.4298553, 155.5868530
39: -118.0546646, 56.9682083, -118.1042862, 56.8845100, -174.9391785, 175.0724945
40: -99.6710968, 35.1010628, -99.6567383, 35.0283737, -134.6994476, 134.7577972
41: -83.7876358, 50.9048996, -83.7897644, 50.8623161, -134.6499481, 134.6946716
42: -65.8793335, 37.7656212, -65.8330688, 37.7202492, -103.5995636, 103.5986938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=454, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 955

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0994136, upper bound: 70.2076412
time: 75.30 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0995067, upper bound: 70.2340678
time: 83.20 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -91.1254730, 65.6169891, -91.0178452, 65.5354767, -156.6609344, 156.6348267
1: -45.4209480, 55.9216156, -45.3806992, 55.8743668, -101.2953033, 101.3023148
2: -39.6539536, 56.9289627, -39.6375046, 56.8688126, -96.5227661, 96.5664673
3: -49.7014122, 58.9306488, -49.6655273, 58.8493233, -108.5507355, 108.5961761
4: -48.5130692, 72.7986908, -48.4632683, 72.6962662, -121.2093353, 121.2619400
5: -45.9317627, 57.8540535, -45.9016953, 57.7717209, -103.7034836, 103.7557449
6: -90.5667725, 43.6714172, -90.4204102, 43.5092773, -134.0760498, 134.0918274
7: -54.6287918, 56.5652008, -54.6330566, 56.5131149, -111.1419067, 111.1982422
8: -60.4310608, 82.4516449, -60.3754616, 82.3741302, -142.8051910, 142.8271027
9: -49.3332787, 63.2666855, -49.1802368, 63.1657028, -112.4989700, 112.4469147
10: -76.3868179, 71.4060898, -76.2035065, 71.3797913, -147.7666016, 147.6095886
11: -80.4024429, 37.1801338, -80.3001709, 37.1526642, -117.5551071, 117.4803009
12: -84.5497742, 50.8312988, -84.4635162, 50.8086700, -135.3584442, 135.2947998
13: -77.3077240, 80.4623871, -77.2676086, 80.3516846, -157.6593933, 157.7299805
14: -117.0583801, 55.1164856, -116.9199753, 55.1315155, -172.1898956, 172.0364685
15: -60.2458687, 62.8946152, -60.0735092, 62.7810440, -123.0269089, 122.9681244
16: -78.9053345, 54.4506035, -78.8199997, 54.4384079, -133.3437347, 133.2705994
17: -110.4787827, 47.4855080, -110.3560486, 47.4887466, -157.9675293, 157.8415527
18: -78.6788559, 54.0558929, -78.6298752, 54.0872536, -132.7661133, 132.6857605
19: -57.4743996, 35.8351440, -57.4070969, 35.8320465, -93.3064423, 93.2422409
20: -56.1843033, 39.5282440, -56.1158562, 39.4817314, -95.6660309, 95.6440887
21: -73.7380219, 41.2001953, -73.6589508, 41.2018814, -114.9399033, 114.8591461
22: -68.8415375, 43.8490257, -68.7466431, 43.8506737, -112.6921997, 112.5956650
23: -61.2997589, 46.4349670, -61.2288704, 46.4237442, -107.7234955, 107.6638336
24: -73.1295624, 46.0320511, -73.0002899, 45.9690857, -119.0986404, 119.0323410
25: -63.9038620, 47.2460785, -63.8072281, 47.1961670, -111.1000290, 111.0533066
26: -82.6424561, 61.3717537, -82.5829010, 61.4386749, -144.0811310, 143.9546509
27: -69.0346832, 45.8126755, -68.9212036, 45.7803345, -114.8149948, 114.7338791
28: -58.0964737, 48.5963669, -57.9963799, 48.5677490, -106.6642227, 106.5927429
29: -74.8817902, 42.0330734, -74.7843018, 42.0493431, -116.9311371, 116.8173752
30: -78.6966934, 47.5723801, -78.5752945, 47.5145531, -126.2112427, 126.1476669
31: -79.8009644, 47.5255699, -79.7136688, 47.5259323, -127.3268967, 127.2392349
32: -83.2052460, 42.5303192, -83.1076202, 42.4346848, -125.6399307, 125.6379395
33: -109.4218216, 51.9366760, -109.3556671, 51.7853470, -161.2071533, 161.2923279
34: -97.4746246, 28.3421021, -97.3943024, 28.2248764, -125.6995010, 125.7363968
35: -91.2382965, 39.5293884, -91.1853561, 39.4179764, -130.6562805, 130.7147522
36: -89.7622528, 45.4264984, -89.7024689, 45.3439407, -135.1062012, 135.1289673
37: -131.0497284, 40.2221413, -130.8868256, 40.0870972, -171.1368256, 171.1089630
38: -106.3816223, 49.4958534, -106.3271713, 49.3464737, -155.7280884, 155.8230286
39: -118.2315445, 57.0329857, -118.1677551, 56.9166946, -175.1482239, 175.2007446
40: -99.8201599, 35.1952896, -99.7085495, 35.0711899, -134.8913422, 134.9038391
41: -83.9048538, 51.0259094, -83.8205490, 50.9247208, -134.8295593, 134.8464661
42: -65.9665070, 37.9105835, -65.8531342, 37.7965050, -103.7630157, 103.7637100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=454, inp2_unstable=455, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=543, inp2_unstable=543, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1748
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 969
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1004
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 968
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1526
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1625
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 789

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1747

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0994136, upper bound: 70.2078148
time: 75.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -70.0995067, upper bound: 70.2342616
time: 86.12 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 163.94 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 163.94
Output dim: 4, lower bound: -70.0994136, upper bound: 70.2076412
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 163.94
Output dim: 4, lower bound: -70.0995067, upper bound: 70.2340678
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 163.94
Output dim: 4, lower bound: -70.0994136, upper bound: 70.2078148
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 163.94
Output dim: 4, lower bound: -70.0995067, upper bound: 70.2342616
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2373597
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2375630
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2372314
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2374416
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2372314
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2374416
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1036525, upper bound: 70.2577376
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1036525, upper bound: 70.2579013
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1409378, upper bound: 70.2577376
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1409378, upper bound: 70.2579013
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1036525, upper bound: 70.2575274
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1036525, upper bound: 70.2576854
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1409378, upper bound: 70.2575274
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2576854
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2373597
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2375630
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2373597
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2375630
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1814861, upper bound: 70.2372314
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2374416
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2372314
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2374416
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2577376
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1776395, upper bound: 70.2579013
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2577376
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1409378, upper bound: 70.2579013
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1027793, upper bound: 70.2575274
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.2205967, upper bound: 70.2576854
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.1401340, upper bound: 70.2575274
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 163.94
Output dim: 4, lower bound: -70.2576850, upper bound: 70.2576854

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 95.99 + 7114.38 = 7210.37 seconds
